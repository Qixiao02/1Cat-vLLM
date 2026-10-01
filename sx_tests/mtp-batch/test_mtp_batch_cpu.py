# SPDX-License-Identifier: Apache-2.0
"""SX MTP batch routes (upstream 1Cat MTP4 kernels): gating, packing, dispatch.

GPU: none. On a bare host the route modules are loaded through
``mtp_batch_boot.py``; inside the serving image the real modules are used.

    python sx_tests/mtp-batch/test_mtp_batch_cpu.py
    python -m pytest -q sx_tests/mtp-batch/test_mtp_batch_cpu.py

Asserted (SX_OPT_MTP_HC_BATCH):
* switch parsing: defaults on, ``SX_OPT_*`` wins over the upstream alias, the
  alias alone works, "0" disables;
* load-time contract: the native-MTP lane target with k = 4 only (no-MTP,
  k != 4, other methods, TP2, micro-batching, batch invariance, dual compile
  off and SX_OPT_MTP_LANE=0 are rejected);
* runtime admission: M5 / M10 only, inside the FULL verify capture of an
  installed lane, under the MTP precision policy;
* packing keeps every FP16 bit and this rank's TP4 ownership; bad roles,
  shapes, dtypes and ranks are rejected;
* the HC loader tags (and moves the up projection to the FP16 method) only
  in the admitted lane; packed copies are made only for tagged layers with
  the native op and a registered communicator (nothing otherwise);
* an admitted batch calls the communicator with the MTP contract
  (FP16 partials, cooperative / full unroll from the switches); no packed
  copy (every non-MTP deployment) never reads the MTP switches;
* the opaque op keeps its fake shapes and forwards both packed buffers.

Asserted (SX_OPT_MTP_ROUTER_BATCH, SX_OPT_MTP_BATCH_OVER_ROWS):
* switch parsing (router default on, OVER_ROWS default off);
* packing keeps every router weight bit, bad geometries are rejected;
* runtime admission: packed copy, CUDA FP16 aligned [M5/M10, 2560] inside
  the verify capture, and precedence: an admitted SX_OPT_ROWS kernel keeps
  its width unless OVER_ROWS=1;
* dispatch: only the router role at (512, 2560) reaches the native op, with
  the rows tile of that call; everything else keeps its old route;
* the loader tags only routers, only in the admitted lane; packing happens
  only for tagged layers with the native op; apply() forwards the pack.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mtp_batch_boot as boot  # noqa: E402

gemv, hc = boot.install()

from vllm import envs  # noqa: E402
from vllm.compilation import sm70_decode_graph as dg  # noqa: E402
from vllm.model_executor.layers.linear import UnquantizedLinearMethod  # noqa: E402

SWITCHES = (
    "SX_OPT_MTP_HC_BATCH",
    "VLLM_SM70_MTP_HC_BATCH",
    "SX_OPT_MTP_HC_COOPERATIVE",
    "VLLM_SM70_MTP_HC_COOPERATIVE",
    "SX_OPT_MTP_HC_FULL_UNROLL",
    "VLLM_SM70_MTP_HC_FULL_UNROLL",
    "SX_OPT_MTP_ROUTER_BATCH",
    "VLLM_SM70_MTP_ROUTER_BATCH",
    "SX_OPT_MTP_BATCH_OVER_ROWS",
)
PRECISION = (
    "allow_fp16_reduced_precision_reduction",
    "allow_bf16_reduced_precision_reduction",
    "allow_fp16_accumulation",
)


def _reset_env_cache() -> None:
    getattr(envs, "disable_envs_cache", lambda: None)()
    gemv._sx_mtp_batch_config.cache_clear()


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for key in SWITCHES:
        monkeypatch.delenv(key, raising=False)
    for key in ("VLLM_BATCH_INVARIANT", "SX_OPT_MTP_LANE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("VLLM_SM70_QWEN38_DUAL_COMPILE", "1")
    matmul = torch.backends.cuda.matmul
    saved = [getattr(matmul, name) for name in PRECISION]
    installed = dg.sm70_mtp_lane_installed()
    _reset_env_cache()
    yield
    for name, value in zip(PRECISION, saved):
        setattr(matmul, name, value)
    dg.set_sm70_mtp_lane_installed(installed)
    _reset_env_cache()


@contextmanager
def _verify_capture(installed: bool = True, compiling: bool = True):
    """The target FULL verify-graph capture of the installed MTP lane."""
    saved = dg.sm70_mtp_lane_installed()
    dg.set_sm70_mtp_lane_installed(installed)
    matmul = torch.backends.cuda.matmul
    matmul.allow_fp16_reduced_precision_reduction = True
    matmul.allow_fp16_accumulation = False
    try:
        with dg.sm70_decode_graph_compilation(compiling):
            yield
    finally:
        dg.set_sm70_mtp_lane_installed(saved)


class _FakeCuda(torch.Tensor):
    """A CPU tensor that reports is_cuda (exercises the GPU-only loaders)."""

    @property
    def is_cuda(self):  # type: ignore[override]
        return True


def _fake_cuda(t: torch.Tensor) -> torch.Tensor:
    return torch.Tensor._make_subclass(_FakeCuda, t)


class _Comm:
    def __init__(self, admit: bool = True, rank: int = 2):
        self.admit = admit
        self.rank = rank
        self.calls: list[tuple[tuple, dict]] = []

    def can_sm70_qwen38_hc_batch(self, x):
        return self.admit and x.ndim == 2 and x.shape[1] == 10240

    def sm70_qwen38_hc_batch(self, *args, **kwargs):
        self.calls.append((args, kwargs))


# ---------------------------------------------------------------------------
# Switches and admission
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, (True, True, True)),
        ({"SX_OPT_MTP_HC_BATCH": "0"}, (False, True, True)),
        ({"VLLM_SM70_MTP_HC_BATCH": "0"}, (False, True, True)),
        ({"SX_OPT_MTP_HC_BATCH": "1", "VLLM_SM70_MTP_HC_BATCH": "0"}, (True, True, True)),
        ({"SX_OPT_MTP_HC_BATCH": " ", "VLLM_SM70_MTP_HC_BATCH": "0"}, (False, True, True)),
        ({"SX_OPT_MTP_HC_COOPERATIVE": "0"}, (True, False, True)),
        ({"VLLM_SM70_MTP_HC_FULL_UNROLL": "0"}, (True, True, False)),
    ],
)
def test_hc_switches(monkeypatch, env, expected):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    _reset_env_cache()
    config = gemv._sx_mtp_batch_config()
    assert (config.hc, config.hc_cooperative, config.hc_full_unroll) == expected


def test_contract_admits_only_the_k4_mtp_lane(monkeypatch):
    assert gemv._sx_mtp_batch_contract(boot.lane_config(4))
    for k in (1, 3, 5, 7):
        assert not gemv._sx_mtp_batch_contract(boot.lane_config(k)), k
    assert not gemv._sx_mtp_batch_contract(boot.lane_config(None))
    for method in ("eagle3", "dflash"):
        assert not gemv._sx_mtp_batch_contract(boot.lane_config(4, method))
    assert not gemv._sx_mtp_batch_contract(boot.lane_config(4, tp=2))
    config = boot.lane_config(4)
    config.parallel_config.use_ubatching = True
    assert not gemv._sx_mtp_batch_contract(config)
    config = boot.lane_config(4)
    config.model_config.hf_text_config.num_hidden_layers = 40
    assert not gemv._sx_mtp_batch_contract(config)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    _reset_env_cache()
    assert not gemv._sx_mtp_batch_contract(boot.lane_config(4))
    monkeypatch.delenv("VLLM_BATCH_INVARIANT")
    monkeypatch.setenv("VLLM_SM70_QWEN38_DUAL_COMPILE", "0")
    _reset_env_cache()
    assert not gemv._sx_mtp_batch_contract(boot.lane_config(4))
    monkeypatch.setenv("VLLM_SM70_QWEN38_DUAL_COMPILE", "1")
    monkeypatch.setenv("SX_OPT_MTP_LANE", "0")
    _reset_env_cache()
    assert not gemv._sx_mtp_batch_contract(boot.lane_config(4))
    # Partial configs fail closed.
    assert not gemv._sx_mtp_batch_contract(SimpleNamespace())


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 5, 8, 10, 15, 16, 20])
def test_rows_admission_is_m5_m10_in_the_verify_capture(rows):
    x = torch.empty(rows, 2560, dtype=torch.float16)
    with _verify_capture():
        assert gemv._sx_mtp_batch_rows_ok(x) == (rows in (5, 10))
    # Outside the capture, without the lane, or inside a no-MTP capture.
    assert not gemv._sx_mtp_batch_rows_ok(x)
    with _verify_capture(compiling=False):
        assert not gemv._sx_mtp_batch_rows_ok(x)
    with _verify_capture(installed=False):
        assert not gemv._sx_mtp_batch_rows_ok(x)


def test_rows_admission_requires_the_mtp_precision_policy(monkeypatch):
    x = torch.empty(5, 2560, dtype=torch.float16)
    matmul = torch.backends.cuda.matmul
    with _verify_capture():
        assert gemv._sx_mtp_batch_rows_ok(x)
        matmul.allow_fp16_reduced_precision_reduction = False
        assert not gemv._sx_mtp_batch_rows_ok(x)
        matmul.allow_fp16_reduced_precision_reduction = True
        matmul.allow_fp16_accumulation = True
        assert not gemv._sx_mtp_batch_rows_ok(x)
        matmul.allow_fp16_accumulation = False
        monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
        _reset_env_cache()
        assert not gemv._sx_mtp_batch_rows_ok(x)
        assert not gemv._sx_mtp_batch_rows_ok(torch.empty(5, 2, 2560))


# ---------------------------------------------------------------------------
# HC packing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rank", range(4))
def test_hc_down_pack_preserves_bits_and_tp_ownership(rank):
    raw = torch.randint(-(2**15), 2**15, (336, 10240), dtype=torch.int16)
    packed = hc._pack_hc_batch_weight(raw.view(torch.float16), "down", rank)
    assert packed.shape == (3, 640, 2, 32, 8)
    restored = packed.permute(0, 3, 1, 2, 4).contiguous().view(96, 10240)
    assert torch.equal(restored[:88].view(torch.int16), raw[rank * 80 : rank * 80 + 88])
    assert not restored[88:].view(torch.int16).count_nonzero()
    if rank == 3:
        assert torch.equal(restored[80:84].view(torch.int16), raw[320:324])


@pytest.mark.parametrize("rank", range(4))
def test_hc_up_pack_preserves_all_branch_bits(rank):
    raw = torch.randint(-(2**15), 2**15, (10240, 320), dtype=torch.int16)
    packed = hc._pack_hc_batch_weight(raw.view(torch.float16), "up", rank)
    assert packed.shape == (80, 20, 2, 4, 8, 8)
    restored = packed.permute(3, 0, 4, 1, 2, 5).contiguous().view(4, 640, 320)
    assert torch.equal(
        restored.view(torch.int16),
        raw.view(4, 2560, 320)[:, rank * 640 : (rank + 1) * 640],
    )


@pytest.mark.parametrize(
    "shape,dtype,role,rank",
    [
        ((336, 10240), torch.float16, "down", -1),
        ((336, 10240), torch.float16, "down", 4),
        ((336, 10240), torch.float32, "down", 0),
        ((336, 10240), torch.float16, "up", 0),
        ((324, 10240), torch.float16, "down", 0),
        ((10240, 320), torch.float16, "other", 0),
    ],
)
def test_hc_bad_packing_rejected(shape, dtype, role, rank):
    with pytest.raises(ValueError):
        hc._pack_hc_batch_weight(torch.empty(shape, dtype=dtype), role, rank)


# ---------------------------------------------------------------------------
# HC loader
# ---------------------------------------------------------------------------
def _hc_module() -> torch.nn.Module:
    module = torch.nn.Module()
    module.use_combine, module.lora_rank = True, 320
    module.hc_count, module.hidden_size = 4, 2560
    for name, shape in (
        ("input_mix_weight_down_block_inject", (336, 10240)),
        ("input_mix_weight_up", (10240, 320)),
    ):
        layer = torch.nn.Module()
        layer.quant_method = UnquantizedLinearMethod()
        layer.weight = torch.nn.Parameter(
            torch.empty(shape, dtype=torch.float16, device="meta"),
            requires_grad=False,
        )
        module.add_module(name, layer)
    return module


@pytest.mark.parametrize(
    "config,switch,tagged",
    [
        (4, None, True),
        (4, "0", False),
        (3, None, False),
        (None, None, False),
    ],
)
def test_hc_loader_tags_only_the_admitted_lane(monkeypatch, config, switch, tagged):
    monkeypatch.setenv("VLLM_SM70_QWEN38_FUSED_HC_FP16", "1")
    monkeypatch.setenv("VLLM_SM70_QWEN4_EXP_ONLINE_QPN8", "0")
    if switch is not None:
        monkeypatch.setenv("SX_OPT_MTP_HC_BATCH", switch)
    monkeypatch.setattr(hc.current_platform, "is_device_capability", lambda _: True)
    _reset_env_cache()
    module = _hc_module()
    hc.enable_qwen38_sm70_fp16_fused_hc(module, torch.float16, boot.lane_config(config))
    assert module._sm70_qwen38_fp16_fused_hc
    for role, layer in zip(("down", "up"), module.children()):
        assert hasattr(layer, "_sm70_qwen38_hc_batch_role") == tagged
        if tagged:
            assert layer._sm70_qwen38_hc_batch_role == role
            assert isinstance(layer.quant_method, gemv.Qwen38SM70FP16LinearMethod)
        else:
            assert type(layer.quant_method) is UnquantizedLinearMethod
    if config is None:
        # The no-MTP lane never reads (or logs) the MTP switches.
        assert gemv._sx_mtp_batch_config.cache_info().misses == 0


def _loaded_layer(role: str, rank_shape: tuple[int, int]) -> torch.nn.Module:
    layer = torch.nn.Module()
    raw = torch.randint(-(2**15), 2**15, rank_shape, dtype=torch.int16)
    layer.weight = torch.nn.Parameter(
        _fake_cuda(raw.view(torch.float16)), requires_grad=False
    )
    layer._sm70_qwen38_hc_batch_role = role
    return layer


@pytest.mark.parametrize(
    "role,shape", [("down", (336, 10240)), ("up", (10240, 320))]
)
def test_hc_process_weights_packs_with_the_communicator_rank(monkeypatch, role, shape):
    comm = _Comm(rank=2)
    monkeypatch.setattr(hc, "_sx_tp4_custom_ar", lambda: comm)
    monkeypatch.setattr(boot, "HC_BATCH_OP", [True])
    if not boot.STUBBED:
        from vllm import _custom_ops as ops

        monkeypatch.setattr(ops, "supports_sm70_qwen38_hc_batch", lambda: True)
    layer = _loaded_layer(role, shape)
    gemv.Qwen38SM70FP16LinearMethod().process_weights_after_loading(layer)
    expected = hc._pack_hc_batch_weight(layer.weight.as_subclass(torch.Tensor), role, 2)
    packed = layer._sm70_qwen38_hc_batch_packed
    assert torch.equal(
        packed.as_subclass(torch.Tensor).view(torch.int16), expected.view(torch.int16)
    )
    assert "_sm70_qwen38_hc_batch_packed" not in layer.state_dict()


@pytest.mark.parametrize("missing", ["op", "comm", "unregistered"])
def test_hc_process_weights_allocates_nothing_without_the_route(monkeypatch, missing):
    comm = None if missing == "comm" else _Comm(admit=missing != "unregistered")
    monkeypatch.setattr(hc, "_sx_tp4_custom_ar", lambda: comm)
    if boot.STUBBED:
        monkeypatch.setattr(boot, "HC_BATCH_OP", [missing != "op"])
    else:
        from vllm import _custom_ops as ops

        monkeypatch.setattr(
            ops, "supports_sm70_qwen38_hc_batch", lambda: missing != "op"
        )
    layer = _loaded_layer("down", (336, 10240))
    gemv.Qwen38SM70FP16LinearMethod().process_weights_after_loading(layer)
    assert not hasattr(layer, "_sm70_qwen38_hc_batch_packed")


def test_untagged_layer_is_untouched(monkeypatch):
    monkeypatch.setattr(hc, "_sx_tp4_custom_ar", lambda: pytest.fail("comm read"))
    layer = _loaded_layer("down", (336, 10240))
    del layer._sm70_qwen38_hc_batch_role
    gemv.Qwen38SM70FP16LinearMethod().process_weights_after_loading(layer)
    assert not hasattr(layer, "_sm70_qwen38_hc_batch_packed")


# ---------------------------------------------------------------------------
# HC dispatch
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "env,cooperative,full_unroll",
    [
        ({}, True, True),
        ({"SX_OPT_MTP_HC_COOPERATIVE": "0"}, False, True),
        ({"SX_OPT_MTP_HC_FULL_UNROLL": "0"}, True, False),
    ],
)
@pytest.mark.parametrize("rows", [5, 10])
def test_admitted_hc_calls_the_communicator_with_the_mtp_contract(
    monkeypatch, env, cooperative, full_unroll, rows
):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    _reset_env_cache()
    comm = _Comm()
    monkeypatch.setattr(hc, "_sx_tp4_custom_ar", lambda: comm)
    monkeypatch.setattr(hc, "_batch_runtime_ok", lambda *args: True)
    x = torch.empty(rows, 10240, dtype=torch.float16)
    down, up = torch.empty(3, 640, 2, 32, 8), torch.empty(80, 20, 2, 4, 8, 8)
    block, injection = hc._qwen38_sm70_fp16_fused_hc(x, x, x, down, up)
    (args, kwargs), = comm.calls
    assert kwargs == {
        "round_down_partials": True,
        "cooperative": cooperative,
        "full_unroll": full_unroll,
    }
    assert args[0] is x and args[1] is down and args[2] is up
    assert args[3].shape == (20, rows, 96) and args[3].dtype == torch.float32
    assert [tuple(t.shape) for t in args[4:]] == [
        (rows, 320),
        (rows, 640),
        (rows, 2560),
        (rows, 4),
    ]
    assert args[6] is block and args[7] is injection


def test_unregistered_communicator_falls_through(monkeypatch):
    monkeypatch.setattr(hc, "_sx_tp4_custom_ar", lambda: _Comm(admit=False))
    x = torch.empty(5, 10240, dtype=torch.float16)
    assert hc._sx_hc_batch_forward(x, x, x) is None
    monkeypatch.setattr(hc, "_sx_tp4_custom_ar", lambda: None)
    assert hc._sx_hc_batch_forward(x, x, x) is None


@pytest.mark.parametrize("rows", [1, 2, 5, 10, 16, 17])
def test_runtime_admission_without_packs_or_on_cpu(rows):
    x = torch.empty(rows, 10240, dtype=torch.float16)
    down, up = torch.empty(3, 640, 2, 32, 8, dtype=torch.float16), torch.empty(
        80, 20, 2, 4, 8, 8, dtype=torch.float16
    )
    with _verify_capture():
        # No packed copy (every deployment outside the admitted lane): the
        # MTP switches are not even read.
        assert not hc._batch_runtime_ok(x, None, None)
        assert gemv._sx_mtp_batch_config.cache_info().misses == 0
        # CPU tensors are never admitted.
        assert not hc._batch_runtime_ok(x, down, up)
    fake = hc._qwen38_sm70_fp16_fused_hc_fake(x, x, x, x, x)
    assert [tuple(t.shape) for t in fake] == [(rows, 2560), (rows, 4)]


def test_maybe_apply_forwards_both_packed_buffers(monkeypatch):
    calls = []

    class _Ops:
        @staticmethod
        def qwen38_sm70_fp16_fused_hc(*args):
            calls.append(args)
            return "out"

    module = _hc_module()
    down = module.input_mix_weight_down_block_inject
    up = module.input_mix_weight_up
    x = torch.empty(5, 10240, dtype=torch.float16, device="meta")
    pd = torch.empty(3, 640, 2, 32, 8, dtype=torch.float16, device="meta")
    pu = torch.empty(80, 20, 2, 4, 8, 8, dtype=torch.float16, device="meta")
    monkeypatch.setattr(hc, "use_sm70_decode_graph_semantics", lambda: True)
    # torch.ops is swapped only around the calls below.
    monkeypatch.setattr(hc.torch, "ops", SimpleNamespace(vllm=_Ops))
    assert hc.maybe_apply_qwen38_sm70_fp16_fused_hc(down, up, x, True) == "out"
    assert calls[-1][3] is None and calls[-1][4] is None
    down.register_buffer("_sm70_qwen38_hc_batch_packed", pd, persistent=False)
    up.register_buffer("_sm70_qwen38_hc_batch_packed", pu, persistent=False)
    assert hc.maybe_apply_qwen38_sm70_fp16_fused_hc(down, up, x, True) == "out"
    assert calls[-1][3] is pd and calls[-1][4] is pu
    assert hc.maybe_apply_qwen38_sm70_fp16_fused_hc(down, up, x, False) is None


def test_registered_op_accepts_the_packed_arguments():
    x = torch.empty(5, 10240, dtype=torch.float16, device="meta")
    down = torch.empty(336, 10240, dtype=torch.float16, device="meta")
    up = torch.empty(10240, 320, dtype=torch.float16, device="meta")
    pd = torch.empty(3, 640, 2, 32, 8, dtype=torch.float16, device="meta")
    pu = torch.empty(80, 20, 2, 4, 8, 8, dtype=torch.float16, device="meta")
    for args in ((x, down, up), (x, down, up, pd, pu), (x, down, up, None, None)):
        block, injection = torch.ops.vllm.qwen38_sm70_fp16_fused_hc(*args)
        assert block.shape == (5, 2560) and injection.shape == (5, 4)


# ---------------------------------------------------------------------------
# Router batch projection
# ---------------------------------------------------------------------------
ROUTER = "model.layers.3.mlp.gate"


class _NativeC:
    """Recording stand-in for torch.ops._C (only the named ops exist)."""

    def __init__(self, *names: str):
        self.calls: list[tuple[str, tuple]] = []
        for name in names:
            setattr(self, name, self._recorder(name))

    def _recorder(self, name):
        def call(*args):
            self.calls.append((name, args))

        return call


def _swap_native(monkeypatch, native: _NativeC, vllm_ops=None) -> None:
    """Replace torch.ops for the module under test (restored by monkeypatch)."""
    monkeypatch.setattr(
        gemv.torch, "ops", SimpleNamespace(_C=native, vllm=vllm_ops or SimpleNamespace())
    )


@pytest.mark.parametrize(
    "env,router,over_rows",
    [
        ({}, True, False),
        ({"SX_OPT_MTP_ROUTER_BATCH": "0"}, False, False),
        ({"VLLM_SM70_MTP_ROUTER_BATCH": "0"}, False, False),
        ({"SX_OPT_MTP_ROUTER_BATCH": "1", "VLLM_SM70_MTP_ROUTER_BATCH": "0"}, True, False),
        ({"SX_OPT_MTP_BATCH_OVER_ROWS": "1"}, True, True),
        ({"SX_OPT_MTP_BATCH_OVER_ROWS": "0"}, True, False),
    ],
)
def test_router_switches(monkeypatch, env, router, over_rows):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    _reset_env_cache()
    config = gemv._sx_mtp_batch_config()
    assert (config.router, config.over_rows) == (router, over_rows)


def test_router_pack_preserves_every_weight_bit():
    raw = torch.randint(-(2**15), 2**15, (512, 2560), dtype=torch.int16)
    packed = gemv._pack_router_batch_weight(raw.view(torch.float16))
    assert packed.shape == (64, 40, 2, 4, 8, 8)
    restored = packed.permute(0, 4, 3, 1, 2, 5).contiguous().view(512, 2560)
    assert torch.equal(restored.view(torch.int16), raw)


@pytest.mark.parametrize(
    "shape,dtype", [((512, 2561), torch.float16), ((511, 2560), torch.float16),
                    ((512, 2560), torch.float32)]
)
def test_router_bad_pack_rejected(shape, dtype):
    with pytest.raises(ValueError):
        gemv._pack_router_batch_weight(torch.empty(shape, dtype=dtype))


@pytest.mark.parametrize(
    "tile,over_rows,admitted", [(0, "0", True), (5, "0", False), (5, "1", True)]
)
def test_router_runtime_admission_and_precedence(monkeypatch, tile, over_rows, admitted):
    monkeypatch.setenv("SX_OPT_MTP_BATCH_OVER_ROWS", over_rows)
    _reset_env_cache()
    x = _fake_cuda(torch.zeros(5, 2560, dtype=torch.float16))
    packed = _fake_cuda(torch.zeros(64, 40, 2, 4, 8, 8, dtype=torch.float16))
    with _verify_capture():
        assert gemv._router_batch_runtime_ok(x, packed, tile) == admitted
        assert not gemv._router_batch_runtime_ok(x, None, tile)
        # CPU tensors, wrong widths and wrong packs are never admitted.
        assert not gemv._router_batch_runtime_ok(x.as_subclass(torch.Tensor), packed)
        assert not gemv._router_batch_runtime_ok(x[:4], packed)
        assert not gemv._router_batch_runtime_ok(x, packed[:32])
        assert not gemv._router_batch_runtime_ok(x, packed.float())
    assert not gemv._router_batch_runtime_ok(x, packed, 0)  # outside the capture
    monkeypatch.setenv("SX_OPT_MTP_ROUTER_BATCH", "0")
    _reset_env_cache()
    with _verify_capture():
        assert not gemv._router_batch_runtime_ok(x, packed, 0)


@pytest.mark.parametrize("admit", [True, False])
@pytest.mark.parametrize("rows_tile", [0, 5])
def test_router_dispatch(monkeypatch, admit, rows_tile):
    seen = []

    def runtime_ok(x, packed, tile):
        seen.append(tile)
        return admit

    monkeypatch.setattr(gemv, "_router_batch_runtime_ok", runtime_ok)
    monkeypatch.setattr(gemv, "_sx_rows_tile", lambda *a: rows_tile)
    rows_calls = []
    monkeypatch.setattr(
        gemv, "_sx_rows_gemv", lambda x, w, plan, tile: rows_calls.append(tile) or "rows"
    )
    native = _NativeC("qwen38_router_batch_sm70_out")
    x = torch.randn(5, 2560).half()
    w = torch.randn(512, 2560).half()
    packed = torch.empty(64, 40, 2, 4, 8, 8, dtype=torch.float16)
    _swap_native(monkeypatch, native)
    out = gemv._qwen38_sm70_fp16_gemv(x, w, ROUTER, packed)
    assert seen == [rows_tile]
    if admit:
        ((name, args),) = native.calls
        assert args[1] is x and args[2] is packed and args[0] is out
        assert out.shape == (5, 512) and out.dtype == torch.float16
    else:
        assert not native.calls
        assert out == "rows" if rows_tile else out.shape == (5, 512)
    # Other roles never consult the router route.
    seen.clear()
    gemv._qwen38_sm70_fp16_gemv(x, torch.randn(640, 2560).half(),
                                "model.layers.3.self_attn.indexer.index_qk_proj", packed)
    assert not seen


class _Dense(torch.nn.Module):
    def __init__(self, prefix: str, shape: tuple[int, int]):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(shape, dtype=torch.float16, device="meta"), requires_grad=False
        )
        self.prefix = prefix
        self.quant_method = UnquantizedLinearMethod()


@pytest.mark.parametrize(
    "config,switch,tagged",
    [(4, None, True), (4, "0", False), (2, None, False), (None, None, False)],
)
def test_router_loader_tags_only_routers_in_the_lane(monkeypatch, config, switch, tagged):
    monkeypatch.setattr(gemv, "LinearBase", _Dense)
    monkeypatch.setattr(gemv.current_platform, "is_device_capability", lambda _: True)
    monkeypatch.setenv("VLLM_SM70_QWEN38_FP16_GEMV", "1")
    monkeypatch.setenv("VLLM_SM70_QWEN4_EXP_ONLINE_QPN8", "0")
    monkeypatch.setenv("VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16", "0")
    if switch is not None:
        monkeypatch.setenv("SX_OPT_MTP_ROUTER_BATCH", switch)
    _reset_env_cache()
    model = torch.nn.Module()
    model.router = _Dense(ROUTER, (512, 2560))
    model.index = _Dense("model.layers.3.self_attn.indexer.index_qk_proj", (640, 2560))
    model.other = _Dense("model.layers.3.mlp.gate", (513, 2560))
    gemv.enable_qwen38_sm70_fp16_gemv(model, torch.float16, boot.lane_config(config))
    assert isinstance(model.router.quant_method, gemv.Qwen38SM70FP16LinearMethod)
    assert getattr(model.router, "_sm70_mtp_prepare_router_batch", False) == tagged
    assert not hasattr(model.index, "_sm70_mtp_prepare_router_batch")
    assert not hasattr(model.other, "_sm70_mtp_prepare_router_batch")
    assert type(model.other.quant_method) is UnquantizedLinearMethod
    if config is None:
        assert gemv._sx_mtp_batch_config.cache_info().misses == 0


@pytest.mark.parametrize("op_present", [True, False])
def test_router_process_weights(monkeypatch, op_present):
    layer = torch.nn.Module()
    raw = torch.randint(-(2**15), 2**15, (512, 2560), dtype=torch.int16)
    layer.weight = torch.nn.Parameter(_fake_cuda(raw.view(torch.float16)),
                                      requires_grad=False)
    layer._sm70_mtp_prepare_router_batch = True
    native = _NativeC(*(["qwen38_router_batch_sm70_out"] if op_present else []))
    _swap_native(monkeypatch, native)
    gemv.Qwen38SM70FP16LinearMethod().process_weights_after_loading(layer)
    assert hasattr(layer, "_sm70_mtp_router_packed") == op_present
    if op_present:
        expected = gemv._pack_router_batch_weight(raw.view(torch.float16))
        assert torch.equal(
            layer._sm70_mtp_router_packed.as_subclass(torch.Tensor).view(torch.int16),
            expected.view(torch.int16),
        )
        assert "_sm70_mtp_router_packed" not in layer.state_dict()
    # Untagged layers: nothing, whatever the op.
    del layer._sm70_mtp_prepare_router_batch
    if op_present:
        del layer._sm70_mtp_router_packed
    gemv.Qwen38SM70FP16LinearMethod().process_weights_after_loading(layer)
    assert not hasattr(layer, "_sm70_mtp_router_packed")


def test_apply_forwards_the_router_pack(monkeypatch):
    calls = []

    class _VllmOps:
        @staticmethod
        def qwen38_sm70_fp16_gemv(*args):
            calls.append(args)
            return "out"

    monkeypatch.setattr(gemv, "use_sm70_decode_graph_semantics", lambda: True)
    layer = _Dense(ROUTER, (512, 2560))
    x = torch.empty(5, 2560, dtype=torch.float16, device="meta")
    packed = torch.empty(64, 40, 2, 4, 8, 8, dtype=torch.float16, device="meta")
    _swap_native(monkeypatch, _NativeC(), SimpleNamespace(
        qwen38_sm70_fp16_gemv=_VllmOps.qwen38_sm70_fp16_gemv))
    method = gemv.Qwen38SM70FP16LinearMethod()
    assert method.apply(layer, x) == "out"
    assert calls[-1][2] == ROUTER and calls[-1][3] is None
    layer.register_buffer("_sm70_mtp_router_packed", packed, persistent=False)
    assert method.apply(layer, x) == "out"
    assert calls[-1][3] is packed


def test_registered_gemv_op_accepts_the_router_pack():
    x = torch.empty(10, 2560, dtype=torch.float16, device="meta")
    w = torch.empty(512, 2560, dtype=torch.float16, device="meta")
    packed = torch.empty(64, 40, 2, 4, 8, 8, dtype=torch.float16, device="meta")
    for args in ((x, w), (x, w, ROUTER), (x, w, ROUTER, packed), (x, w, ROUTER, None)):
        assert torch.ops.vllm.qwen38_sm70_fp16_gemv(*args).shape == (10, 512)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
