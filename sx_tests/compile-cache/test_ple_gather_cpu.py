# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the PLE gather-by-name route (SX_OPT_COMPILE_CACHE).

    python -m pytest sx_tests/compile-cache/test_ple_gather_cpu.py -q

The pinned-host PLE embedding passed the table address to
``qwen4_exp_ple_pinned_gather`` as a Python int. Inductor writes such an int
into the compiled graph, so a graph reloaded in another process gathered from
the address of a buffer that no longer exists (upstream 5f668ebb9, #622). With
the switch on the module registers itself under its prefix and the new
``qwen4_exp_ple_pinned_gather_by_name`` op resolves the address when it runs.
Checked here, on the real source cut out of ``ple_layer.py``:

  * switch off: ``embedding_lookup`` issues exactly the op calls of BASE_REV;
  * switch on: only the layer name and a host/device flag reach the graph (no
    int address), and the gathered rows are identical to the pointer variant
    for host-only, device-only and split tables;
  * the registration in ``__init__`` happens only with the switch on and
    rejects a duplicate name.
"""

from __future__ import annotations

import os
import sys
import textwrap
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import cache_boot as boot  # noqa: E402

sx = boot.load_sx_module()
boot.silence_sx_logger()

CLS = "Qwen4ExpPinnedHostEmbedding"
DIM = 8


class _TorchProxy:
    """torch with a CPU-safe accelerator and a recording ``ops.vllm``."""

    def __init__(self, vllm_ops):
        self.ops = types.SimpleNamespace(vllm=vllm_ops)
        self.accelerator = types.SimpleNamespace(current_device_index=lambda: 0)

    def __getattr__(self, name):
        return getattr(torch, name)


class _Kernel:
    """Stands in for the Triton gather kernel: rows come from a pointer table."""

    def __init__(self, tables):
        self.tables = tables  # pointer -> [rows, DIM] float tensor
        self.launches = []

    def __getitem__(self, grid):
        def launch(weight_ptr, input_ids, weight_scale, output, **kwargs):
            self.launches.append(weight_ptr)
            table = self.tables[int(weight_ptr)]
            rows = table[input_ids.long()]
            output.copy_(rows * weight_scale.reshape(-1)[0])

        return launch


def _make_world(device_rows: int, host_rows: int, tables_seed: int = 0):
    gen = torch.Generator().manual_seed(tables_seed)
    device_table = torch.randn(max(device_rows, 1), DIM, generator=gen)
    host_table = torch.randn(max(host_rows, 1), DIM, generator=gen)
    device_ptr, host_ptr = 0x1000, 0x2000
    kernel = _Kernel({device_ptr: device_table, host_ptr: host_table})
    layer = types.SimpleNamespace(
        _accelerator_weight_ptrs={0: host_ptr},
        _device_table_ptr=device_ptr,
    )
    return kernel, layer, device_ptr, host_ptr


def _namespace(source: str, kernel: _Kernel, layer, vllm_ops, layer_name: str):
    ns = {
        "torch": _TorchProxy(vllm_ops),
        "triton": types.SimpleNamespace(
            next_power_of_2=lambda n: 1 << (int(n) - 1).bit_length()
        ),
        "_gather_ple_fp8_from_pinned_kernel": kernel,
        "get_forward_context": lambda: types.SimpleNamespace(
            no_compile_layers={layer_name: layer}
        ),
    }
    boot.exec_functions(
        source,
        [
            "embedding_lookup",
            "_embedding_lookup_by_name",
        ]
        if "_embedding_lookup_by_name" in source
        else ["embedding_lookup"],
        ns,
        cls=CLS,
    )
    boot.exec_functions(
        source,
        ["qwen4_exp_ple_pinned_gather"]
        + (
            ["qwen4_exp_ple_pinned_gather_by_name"]
            if "def qwen4_exp_ple_pinned_gather_by_name" in source
            else []
        ),
        ns,
    )
    return ns


class _Recorder:
    """torch.ops.vllm: the two gather ops, recording what the graph carries."""

    def __init__(self, ns_getter):
        self.calls = []
        self._ns = ns_getter

    def qwen4_exp_ple_pinned_gather(self, ids, out, scale, ptr, dim):
        self.calls.append(("ptr", tuple(ids.shape), int(ptr), int(dim)))
        self._ns()["qwen4_exp_ple_pinned_gather"](ids, out, scale, ptr, dim)

    def qwen4_exp_ple_pinned_gather_by_name(
        self, ids, out, scale, name, use_host, dim
    ):
        self.calls.append(("name", tuple(ids.shape), name, bool(use_host), int(dim)))
        self._ns()["qwen4_exp_ple_pinned_gather_by_name"](
            ids, out, scale, name, use_host, dim
        )


def _lookup(source, device_rows, host_rows, by_name, ids, layer_name="m.ple.emb"):
    kernel, layer, device_ptr, host_ptr = _make_world(device_rows, host_rows)
    holder = {}
    ops = _Recorder(lambda: holder["ns"])
    ns = _namespace(source, kernel, layer, ops, layer_name)
    holder["ns"] = ns
    scale = torch.tensor([0.5])
    module = types.SimpleNamespace(
        _accelerator_weight_ptrs={0: host_ptr},
        _device_table_ptr=device_ptr,
        _output_dtype=torch.float32,
        _host_rows=host_rows,
        _device_rows=device_rows,
        embedding_dim=DIM,
        weight_scale=scale,
        layer_name=layer_name,
        _sx_gather_by_name=by_name,
        get_accelerator_weight=lambda device: None,
    )
    module.embedding_lookup = types.MethodType(ns["embedding_lookup"], module)
    if "_embedding_lookup_by_name" in ns:
        module._embedding_lookup_by_name = types.MethodType(
            ns["_embedding_lookup_by_name"], module
        )
    # the op resolves the module by layer name: give it the same attributes
    layer._accelerator_weight_ptrs = module._accelerator_weight_ptrs
    layer._device_table_ptr = module._device_table_ptr
    return module.embedding_lookup(ids), ops.calls


def _current_source():
    return boot.read(boot.PLE)


def _base_source():
    source = boot.git_show(boot.BASE_REV, "vllm/models/qwen4_exp/nvidia/ple_layer.py")
    if source is None:
        pytest.skip("git history with the base revision is not available")
    return source


SPLITS = [(0, 10), (10, 0), (6, 10), (3, 4)]


@pytest.mark.parametrize("split", SPLITS)
def test_switch_off_issues_the_base_op_calls(split):
    device_rows, host_rows = split
    total = device_rows + host_rows
    ids = torch.tensor([0, total - 1, total // 2, 1, total - 1], dtype=torch.int64)
    old_out, old_calls = _lookup(_base_source(), device_rows, host_rows, False, ids)
    new_out, new_calls = _lookup(_current_source(), device_rows, host_rows, False, ids)
    assert new_calls == old_calls
    assert all(call[0] == "ptr" for call in new_calls)
    assert torch.equal(new_out, old_out)


@pytest.mark.parametrize("split", SPLITS)
def test_by_name_gathers_the_same_rows_without_pointers(split):
    device_rows, host_rows = split
    total = device_rows + host_rows
    ids = torch.tensor(
        [0, total - 1, total // 2, 1, total - 1, 0, 2], dtype=torch.int64
    )
    ptr_out, ptr_calls = _lookup(_current_source(), device_rows, host_rows, False, ids)
    name_out, name_calls = _lookup(_current_source(), device_rows, host_rows, True, ids)
    assert torch.equal(name_out, ptr_out)
    assert name_calls, "the by-name path must issue gathers"
    for call in name_calls:
        assert call[0] == "name"
        # the op arguments the compiler sees: ids shape, layer name, flag, dim
        assert isinstance(call[2], str) and isinstance(call[3], bool)
    if host_rows == 0:
        assert [c[3] for c in name_calls] == [False]
    elif device_rows == 0:
        assert [c[3] for c in name_calls] == [True]
    else:
        assert [c[3] for c in name_calls] == [False, True]


def test_by_name_op_reads_the_pointer_at_run_time():
    """The same traced call gathers from a table that moved between processes."""
    source = _current_source()
    kernel, layer, device_ptr, host_ptr = _make_world(4, 6)
    holder = {}
    ns = _namespace(source, kernel, layer, _Recorder(lambda: holder["ns"]), "L")
    holder["ns"] = ns
    op = ns["qwen4_exp_ple_pinned_gather_by_name"]
    ids = torch.tensor([1, 2], dtype=torch.int64)
    scale = torch.tensor([1.0])
    out = torch.empty(2, DIM)
    op(ids, out, scale, "L", True, DIM)
    assert kernel.launches == [host_ptr]
    # "another process": the host table now lives at a different address
    new_ptr = 0x9000
    kernel.tables[new_ptr] = kernel.tables[host_ptr].clone() * 2
    layer._accelerator_weight_ptrs = {0: new_ptr}
    out2 = torch.empty(2, DIM)
    op(ids, out2, scale, "L", True, DIM)
    assert kernel.launches[-1] == new_ptr
    assert torch.equal(out2, out * 2)
    # the device half follows _device_table_ptr the same way
    layer._device_table_ptr = new_ptr
    op(ids, out2, scale, "L", False, DIM)
    assert kernel.launches[-1] == new_ptr
    # an empty batch launches nothing
    n = len(kernel.launches)
    op(ids[:0], out2[:0], scale, "L", True, DIM)
    assert len(kernel.launches) == n


def test_by_name_op_is_registered_with_a_fake_impl():
    source = _current_source()
    assert 'op_name="qwen4_exp_ple_pinned_gather_by_name"' in source
    assert "fake_impl=qwen4_exp_ple_pinned_gather_by_name_fake" in source
    # the fake takes the same arguments as the op (a mismatch breaks tracing)
    import inspect

    ns = {}
    boot.exec_functions(
        source,
        ["qwen4_exp_ple_pinned_gather_by_name", "qwen4_exp_ple_pinned_gather_by_name_fake"],
        ns,
    )
    real = inspect.signature(ns["qwen4_exp_ple_pinned_gather_by_name"])
    fake = inspect.signature(ns["qwen4_exp_ple_pinned_gather_by_name_fake"])
    assert list(real.parameters) == list(fake.parameters)


# --------------------------------------------------------------------------
# registration in __init__
# --------------------------------------------------------------------------


def _init_block() -> str:
    source = _current_source()
    lines = source.split("\n")
    first = next(
        i for i, line in enumerate(lines) if "self._sx_gather_by_name = " in line
    )
    last = next(
        i
        for i, line in enumerate(lines)
        if i > first and "static_forward_context[prefix] = self" in line
    )
    return textwrap.dedent("\n".join(lines[first : last + 1]))


def _run_init(switch: str | None, existing=()):
    context = {name: object() for name in existing}
    config = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(static_forward_context=context)
    )
    module = types.SimpleNamespace()
    saved = dict(os.environ)
    try:
        os.environ.pop("SX_OPT_COMPILE_CACHE", None)
        if switch is not None:
            os.environ["SX_OPT_COMPILE_CACHE"] = switch
        exec(  # noqa: S102
            compile(_init_block(), "<init>", "exec"),
            {
                "self": module,
                "prefix": "m.ple.emb",
                "sx_compile_cache": sx,
                "get_current_vllm_config": lambda: config,
            },
        )
    finally:
        os.environ.clear()
        os.environ.update(saved)
    return module, context


@pytest.mark.parametrize("switch", [None, "0", "off"])
def test_off_does_not_register_or_flag(switch):
    module, context = _run_init(switch)
    assert module._sx_gather_by_name is False
    assert context == {}
    assert not hasattr(module, "layer_name")


@pytest.mark.parametrize("switch", ["1", "subgraph", "aot"])
def test_on_registers_under_the_prefix_once(switch):
    module, context = _run_init(switch)
    assert module._sx_gather_by_name is True
    assert module.layer_name == "m.ple.emb"
    assert context == {"m.ple.emb": module}
    with pytest.raises(ValueError, match="Duplicate layer name"):
        _run_init(switch, existing=("m.ple.emb",))
