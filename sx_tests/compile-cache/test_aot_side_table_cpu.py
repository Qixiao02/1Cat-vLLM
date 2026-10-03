# SPDX-License-Identifier: Apache-2.0
"""CPU test of the Triton side-table save/restore of the AOT serializer (#675).

    python -m pytest sx_tests/compile-cache/test_aot_side_table_cpu.py -q

The upstream test (tests/compile/test_aot_triton_side_table.py) needs a vLLM
install with Triton. This one cuts the three helpers out of
``vllm/compilation/caching.py`` and runs them with torch's real
``kernel_side_table`` and FX graphs against fake Triton kernel objects, so the
index remapping can be checked on any machine with a recent torch.
"""

from __future__ import annotations

import copy
import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
import cache_boot as boot  # noqa: E402

wrap = pytest.importorskip("torch._higher_order_ops.triton_kernel_wrap")


# ---- fake Triton ---------------------------------------------------------


class Config:
    def __init__(self, kwargs, num_warps=4, num_stages=2):
        self.kwargs = kwargs
        self.num_warps = num_warps
        self.num_stages = num_stages


class JITFunction:
    def __init__(self, fn):
        self.fn = fn
        self.__module__ = fn.__module__
        self.__name__ = fn.__name__


class Autotuner:
    def __init__(self, fn, configs, keys):
        self.fn = fn
        self.configs = configs
        self.keys = keys
        self.reset_to_zero = []
        self.restore_value = []
        self.user_defined_pre_hook = None
        self.user_defined_post_hook = None
        self.perf_model = None
        self.early_config_prune = None


def _autotune(configs, key):
    return lambda kernel: Autotuner(kernel, configs, key)


fake_triton = types.SimpleNamespace(
    runtime=types.SimpleNamespace(autotuner=types.SimpleNamespace(Autotuner=Autotuner)),
    autotune=_autotune,
)


def _raw(name):
    def fn():
        pass

    fn.__name__ = name
    fn.__module__ = __name__
    return fn


copy_kernel = JITFunction(_raw("copy_kernel"))
other_kernel = JITFunction(_raw("other_kernel"))


@pytest.fixture
def helpers(monkeypatch):
    table = wrap.KernelSideTable()
    table.reset_table()
    monkeypatch.setattr(wrap, "kernel_side_table", table)
    triton_utils = types.ModuleType("vllm.triton_utils")
    triton_utils.triton = fake_triton
    monkeypatch.setitem(sys.modules, "vllm.triton_utils", triton_utils)
    import importlib
    from collections.abc import Iterator

    ns = {
        "torch": torch,
        "importlib": importlib,
        "Iterator": Iterator,
    }
    boot.exec_functions(
        boot.read(boot.CACHING),
        [
            "_triton_kernel_nodes",
            "_serialize_triton_side_table",
            "_restore_triton_side_table",
        ],
        ns,
    )
    return table, ns


def _graph(table, wrapped=False, kernel=copy_kernel, block=16):
    if wrapped:
        kernel = _autotune([Config({}, num_warps=8, num_stages=3)], [])(kernel)
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.call_function(
        wrap.triton_kernel_wrapper_mutation,
        kwargs={
            "kernel_idx": table.add_kernel(kernel),
            "constant_args_idx": table.add_constant_args({"BLOCK": block}),
            "grid": [(1, 1, 1)],
            "kwargs": {"x": x, "y": x},
            "tma_descriptor_metadata": {},
        },
    )
    graph.output(x)
    return torch.fx.GraphModule({}, graph)


def _node(gm):
    return next(n for n in gm.graph.nodes if n.op == "call_function")


@pytest.mark.parametrize("populated", [False, True])
@pytest.mark.parametrize("wrapped", [False, True])
def test_graph_reload_remaps_indices(helpers, populated, wrapped):
    table, ns = helpers
    gm = _graph(table, wrapped)
    # an unrelated, unimportable entry must not prevent saving this graph
    table.add_kernel(object())
    saved = ns["_serialize_triton_side_table"](gm)
    fresh = copy.deepcopy(gm)  # what GraphPickler would give the new process
    table.reset_table()  # the new process
    if populated:
        other_id = table.add_kernel(other_kernel)
        other_args = table.add_constant_args({"BLOCK": 32})
    ns["_restore_triton_side_table"](fresh, saved)
    node = _node(fresh)
    kernel = table.get_kernel(node.kwargs["kernel_idx"])
    if wrapped:
        assert kernel.fn is copy_kernel
        assert kernel.configs[0].num_warps == 8
        assert kernel.configs[0].num_stages == 3
    else:
        assert kernel is copy_kernel
    assert table.get_constant_args(node.kwargs["constant_args_idx"]) == {"BLOCK": 16}
    if populated:
        # the process's own entries are untouched and the graph does not alias them
        assert table.get_kernel(other_id) is other_kernel
        assert table.get_constant_args(other_args) == {"BLOCK": 32}
        assert node.kwargs["kernel_idx"] != other_id
        assert node.kwargs["constant_args_idx"] != other_args
    # the recompiled code carries the new indices, not the saved ones
    assert f"kernel_idx = {node.kwargs['kernel_idx']}" in fresh.code
    # saving the reloaded graph again records the remapped indices
    again = ns["_serialize_triton_side_table"](fresh)
    table.reset_table()
    reloaded = copy.deepcopy(fresh)
    ns["_restore_triton_side_table"](reloaded, again)
    kernel = table.get_kernel(_node(reloaded).kwargs["kernel_idx"])
    assert (kernel.fn if wrapped else kernel) is copy_kernel


def test_graph_without_triton_nodes_needs_no_table(helpers):
    table, ns = helpers
    gm = torch.fx.symbolic_trace(torch.nn.ReLU())
    ns["_restore_triton_side_table"](gm, None)  # no error
    saved = ns["_serialize_triton_side_table"](gm)
    assert saved == {"kernels": {}, "constants": {}}


def test_legacy_artifact_with_triton_nodes_requires_regeneration(helpers):
    table, ns = helpers
    gm = _graph(table)
    with pytest.raises(RuntimeError, match="no Triton side table"):
        ns["_restore_triton_side_table"](copy.deepcopy(gm), None)


def test_unimportable_referenced_kernel_fails_the_save_loudly(helpers):
    table, ns = helpers
    local = JITFunction(_raw("not_a_module_attribute"))
    gm = _graph(table, kernel=local)
    with pytest.raises(RuntimeError, match="not importable"):
        ns["_serialize_triton_side_table"](gm)


def test_frozen_files_do_not_change_the_backend_code_hash(tmp_path):
    """backends._compute_backend_code_hash: <frozen os> must not matter (on)."""
    import hashlib

    ns = {
        "hashlib": hashlib,
        "Sequence": list,
        "logger": boot.RecordingLogger(),
    }
    boot.exec_functions(boot.read(boot.BACKENDS), ["_compute_backend_code_hash"], ns)
    fn = ns["_compute_backend_code_hash"]
    source = tmp_path / "model.py"
    source.write_text("def forward(x): return x + 1\n")
    files = [str(source)]
    with_frozen = ["<frozen os>", str(source), "<string>"]
    assert fn(files) == fn(with_frozen)  # default: skip frozen names
    assert fn(files, skip_frozen=True) == fn(with_frozen, skip_frozen=True)
    # switch off keeps the pre-switch key, which hashed the frozen names
    assert fn(files, skip_frozen=False) != fn(with_frozen, skip_frozen=False)
    before = fn(files)
    source.write_text("def forward(x): return x + 2\n")
    assert fn(files) != before  # real files still count
    assert fn(["<frozen os>", str(source)]) != before
