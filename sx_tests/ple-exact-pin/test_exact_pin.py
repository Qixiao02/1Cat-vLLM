# SPDX-License-Identifier: Apache-2.0
"""PLE host table pinned at its exact size (SX_OPT_PLE_EXACT_PIN).

Run (CPU is enough, no vLLM install and no GPU needed; see ``pin_boot.py``):

    python sx_tests/ple-exact-pin/test_exact_pin.py
    python -m pytest -q sx_tests/ple-exact-pin/test_exact_pin.py

The CUDA calls are replaced by recorders; everything else is the real code of
``vllm/models/qwen4_exp/nvidia/exact_pin.py`` and of ``materialize_tables``.
What a GPU adds (cuMemHostRegister, ``Tensor.is_pinned``, the UVA view, the
MemAvailable saving) is covered by ``probe_pinned_memory.py``.

Asserted:
* the switch: on by default, off for 0/false/off/no, anything else on;
* the allocation: page aligned, exactly the requested bytes (the mapping is
  the next whole page), zero filled, registered in full at the tensor's own
  address, readable and writable through that address;
* teardown: nothing is unregistered while any view is alive; once the last one
  goes, the registration is dropped exactly once and before the mapping is
  unmapped;
* failure: when registration raises, when torch does not call the registered
  memory pinned, or when the mapping cannot be made, the old
  ``torch.empty(pin_memory=True)`` allocation is returned, a warning is logged
  once, and nothing stays registered;
* ``SX_OPT_PLE_EXACT_PIN=0`` and an empty table use the old call unchanged;
* through the real ``materialize_tables``: the published host table is the
  exact one (or the old one), the available-memory pre-check still refuses, and
  a host allocation failure leaves the layer retryable.
Expected: all pass (the GPU-less fallback case is skipped where CUDA exists).
"""

from __future__ import annotations

import ctypes
import gc
import mmap
import os
import sys
import types

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pin_boot  # noqa: E402

exact_pin = pin_boot.load_exact_pin()
PAGE = mmap.PAGESIZE
FP8 = torch.float8_e4m3fn


def _round_up(value: int) -> int:
    return -(-value // PAGE) * PAGE


class Recorder:
    """Stands in for the CUDA driver and torch's pin check."""

    def __init__(self) -> None:
        self.registered: list[tuple[int, int, int]] = []
        self.unregistered: list[tuple[int, int]] = []
        self.events: list[str] = []
        self.register_error: Exception | None = None
        self.pinned = True
        self.warnings: list[str] = []
        self.infos: list[str] = []


class _RecordingLogger:
    def __init__(self, recorder: Recorder) -> None:
        self._recorder = recorder

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self._recorder.warnings.append(message % args if args else message)

    def info(self, message: str, *args: object, **kwargs: object) -> None:
        self._recorder.infos.append(message % args if args else message)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()

    def register(address: int, nbytes: int, device_index: int) -> None:
        recorder.events.append("register")
        if recorder.register_error is not None:
            raise recorder.register_error
        recorder.registered.append((address, nbytes, device_index))

    def unregister(address: int, device_index: int) -> None:
        recorder.events.append("unregister")
        recorder.unregistered.append((address, device_index))

    monkeypatch.setattr(exact_pin, "_device_index", lambda: 3)
    monkeypatch.setattr(exact_pin, "_host_register", register)
    monkeypatch.setattr(exact_pin, "_host_unregister", unregister)
    monkeypatch.setattr(exact_pin, "_is_pinned", lambda tensor: recorder.pinned)
    monkeypatch.setattr(exact_pin, "_fallback_warned", False)
    monkeypatch.setattr(exact_pin, "logger", _RecordingLogger(recorder))
    monkeypatch.delenv(exact_pin.EXACT_PIN_ENV, raising=False)
    return recorder


class _LegacyCalls:
    """Replaces ``torch.empty`` to see (and survive) the pin_memory=True call."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.calls: list[tuple[tuple, dict]] = []
        real_empty = torch.empty
        self.error: Exception | None = None

        def empty(*args, **kwargs):
            if "pin_memory" in kwargs:
                self.calls.append((args, dict(kwargs)))
                if self.error is not None:
                    raise self.error
                kwargs = {k: v for k, v in kwargs.items() if k != "pin_memory"}
            return real_empty(*args, **kwargs)

        monkeypatch.setattr(torch, "empty", empty)


@pytest.fixture
def legacy(monkeypatch: pytest.MonkeyPatch) -> _LegacyCalls:
    return _LegacyCalls(monkeypatch)


# --- the switch -------------------------------------------------------------


def test_exact_pin_is_on_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(exact_pin.EXACT_PIN_ENV, raising=False)
    assert exact_pin.EXACT_PIN_ENV == "SX_OPT_PLE_EXACT_PIN"
    assert exact_pin.exact_pin_enabled() is True


@pytest.mark.parametrize("value", ["1", " 1 ", "", "2", "true", "yes", "on", "x"])
def test_switch_is_on_for_everything_but_the_off_words(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(exact_pin.EXACT_PIN_ENV, value)
    assert exact_pin.exact_pin_enabled() is True


@pytest.mark.parametrize("value", ["0", " 0 ", "false", "False", "OFF", "no", "No"])
def test_switch_is_off_for_zero_and_the_off_words(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(exact_pin.EXACT_PIN_ENV, value)
    assert exact_pin.exact_pin_enabled() is False


# --- the allocation ---------------------------------------------------------


@pytest.mark.parametrize(
    ("shape", "dtype"),
    [
        ((1, 1), FP8),
        ((3, 13), FP8),
        ((8, 512), FP8),  # exactly one page
        ((8, 513), FP8),  # one byte over a page
        ((1, PAGE - 1), FP8),
        ((PAGE + 1, 1), FP8),
        ((100003, 48), FP8),  # a few MiB, not a page multiple
        ((7, 33), torch.float16),
        ((5, 3), torch.float32),
    ],
)
def test_allocation_is_page_aligned_and_exactly_sized(fake, shape, dtype) -> None:
    nbytes = shape[0] * shape[1] * torch.empty((), dtype=dtype).element_size()

    table = exact_pin.allocate_exact_pinned(shape, dtype)

    assert tuple(table.shape) == shape
    assert table.dtype == dtype
    assert table.device.type == "cpu"
    assert table.is_contiguous()
    assert table.data_ptr() % PAGE == 0
    assert table.numel() * table.element_size() == nbytes
    assert table.untyped_storage().nbytes() == nbytes
    # Registered once, from the tensor's own address, over the whole mapping:
    # the table's bytes rounded up to a page and not a byte more.
    assert fake.registered == [(table.data_ptr(), _round_up(nbytes), 3)]
    assert 0 <= fake.registered[0][1] - nbytes < PAGE
    assert fake.unregistered == []


def test_allocation_is_zero_filled_and_reads_back_what_was_written(fake) -> None:
    rows, dim = 1031, 40
    table = exact_pin.allocate_exact_pinned((rows, dim), FP8)
    raw = table.view(torch.uint8)
    assert not raw.any()

    pattern = (torch.arange(rows * dim) * 7 % 251).to(torch.uint8).reshape(rows, dim)
    raw.copy_(pattern)
    # Straight through the address, not through the tensor.
    seen = ctypes.string_at(table.data_ptr(), rows * dim)
    assert seen == pattern.numpy().tobytes()
    # The way the checkpoint rows are copied in: a narrowed range of rows.
    source = (pattern[5:9] ^ 0xFF).view(FP8)
    table.narrow(0, 5, 4).copy_(source)
    assert torch.equal(raw[5:9], pattern[5:9] ^ 0xFF)
    assert torch.equal(raw[:5], pattern[:5])
    assert torch.equal(raw[9:], pattern[9:])


def test_the_table_of_the_real_workload_is_within_a_page_of_its_request() -> None:
    # 11.92 GiB per rank (VLLM_QWEN4EXP_PLE_HOST_GIB=12 -> 12 GiB minus the
    # rows' remainder) against what the caching host allocator pins.
    requested = int(11.92 * 1024**3)
    assert exact_pin.caching_allocator_bytes(requested) == 16 * 1024**3
    assert _round_up(requested) - requested < PAGE
    assert exact_pin.caching_allocator_bytes(1 << 20) == 1 << 20
    assert exact_pin.caching_allocator_bytes((1 << 20) + 1) == 1 << 21
    assert exact_pin.caching_allocator_bytes(1) == 1


def test_an_empty_table_cannot_be_pinned_exactly(fake) -> None:
    with pytest.raises(ValueError):
        exact_pin.allocate_exact_pinned((0, 16), FP8)
    assert fake.registered == []


# --- teardown ---------------------------------------------------------------


def _collect() -> None:
    gc.collect()
    gc.collect()


def test_registration_is_dropped_once_the_last_view_is_gone(fake) -> None:
    table = exact_pin.allocate_exact_pinned((64, 48), FP8)
    address = table.data_ptr()
    # The UVA view keeps the CPU tensor (so its storage) alive; a slice of it
    # does the same here.
    keeper = table.view(torch.uint8)[10:20]
    del table
    _collect()
    assert fake.unregistered == []
    del keeper
    _collect()
    assert fake.unregistered == [(address, 3)]
    _collect()
    assert len(fake.unregistered) == 1


def test_unregistration_precedes_unmapping(
    fake, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RecordingMap(mmap.mmap):
        def __del__(self) -> None:  # runs right before the base class unmaps
            fake.events.append("unmap")

    monkeypatch.setattr(exact_pin.mmap, "mmap", RecordingMap)
    table = exact_pin.allocate_exact_pinned((16, 16), FP8)
    del table
    _collect()
    assert fake.events == ["register", "unregister", "unmap"]


def test_nothing_is_unregistered_at_interpreter_exit(
    fake, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = []
    real_finalize = exact_pin.weakref.finalize

    class Spy(real_finalize):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            made.append(self)

    monkeypatch.setattr(exact_pin.weakref, "finalize", Spy)
    table = exact_pin.allocate_exact_pinned((16, 16), FP8)
    # At exit the driver reclaims the registration with the context; a
    # finalizer running then would call into a half torn-down CUDA.
    assert len(made) == 1 and made[0].alive and made[0].atexit is False
    del table
    _collect()
    assert not made[0].alive


def test_a_failed_unregistration_is_logged_not_raised(
    fake, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_unregister(address: int, device_index: int) -> None:
        raise RuntimeError("cuMemHostUnregister failed: CUDA_ERROR_INVALID_VALUE")

    monkeypatch.setattr(exact_pin, "_host_unregister", failing_unregister)
    table = exact_pin.allocate_exact_pinned((16, 16), FP8)
    del table
    _collect()
    assert any("cuMemHostUnregister failed" in line for line in fake.warnings)


# --- the fallbacks ----------------------------------------------------------


def test_registration_failure_falls_back_to_pin_memory_and_warns_once(
    fake, legacy
) -> None:
    fake.register_error = RuntimeError(
        "cuMemHostRegister failed: CUDA_ERROR_OUT_OF_MEMORY"
    )

    first = exact_pin.allocate_host_table((6, 8), FP8)
    second = exact_pin.allocate_host_table((6, 8), FP8)

    assert legacy.calls == [
        (((6, 8),), {"dtype": FP8, "device": "cpu", "pin_memory": True})
    ] * 2
    assert tuple(first.shape) == tuple(second.shape) == (6, 8)
    assert first.dtype == FP8
    assert len(fake.warnings) == 1
    assert "falling back to torch.empty(pin_memory=True)" in fake.warnings[0]
    assert "CUDA_ERROR_OUT_OF_MEMORY" in fake.warnings[0]
    assert fake.infos == []
    assert fake.unregistered == []  # nothing was registered, nothing to drop


def test_memory_torch_does_not_call_pinned_is_unregistered_and_replaced(
    fake, legacy
) -> None:
    fake.pinned = False

    table = exact_pin.allocate_host_table((6, 8), FP8)

    assert len(legacy.calls) == 1 and legacy.calls[0][1]["pin_memory"] is True
    assert tuple(table.shape) == (6, 8)
    assert len(fake.registered) == 1
    assert fake.unregistered == [(fake.registered[0][0], 3)]
    assert len(fake.warnings) == 1 and "not report" in fake.warnings[0]


def test_a_failed_mapping_falls_back(
    fake, legacy, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_memory(length: int):
        raise MemoryError("cannot allocate memory")

    monkeypatch.setattr(exact_pin, "_anonymous_mapping", no_memory)
    table = exact_pin.allocate_host_table((6, 8), FP8)

    assert len(legacy.calls) == 1 and tuple(table.shape) == (6, 8)
    assert fake.registered == []
    assert len(fake.warnings) == 1 and "MemoryError" in fake.warnings[0]


@pytest.mark.skipif(
    torch.cuda.is_available(), reason="needs a machine where CUDA is unusable"
)
def test_without_a_working_cuda_the_real_hooks_fall_back(
    monkeypatch: pytest.MonkeyPatch, legacy
) -> None:
    recorder = Recorder()
    monkeypatch.setattr(exact_pin, "_fallback_warned", False)
    monkeypatch.setattr(exact_pin, "logger", _RecordingLogger(recorder))
    monkeypatch.delenv(exact_pin.EXACT_PIN_ENV, raising=False)

    table = exact_pin.allocate_host_table((4, 8), FP8)

    assert len(legacy.calls) == 1 and tuple(table.shape) == (4, 8)
    assert len(recorder.warnings) == 1


def test_the_fallback_failing_too_raises_the_old_error(fake, legacy) -> None:
    fake.register_error = RuntimeError("registration refused")
    legacy.error = RuntimeError("injected pinned allocation failure")
    with pytest.raises(RuntimeError, match="injected pinned allocation failure"):
        exact_pin.allocate_host_table((4, 8), FP8)


def test_switch_off_is_the_old_allocation_unchanged(
    fake, legacy, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(exact_pin.EXACT_PIN_ENV, "0")

    def never(length: int):
        raise AssertionError("the exact path must not run with the switch off")

    monkeypatch.setattr(exact_pin, "_anonymous_mapping", never)
    table = exact_pin.allocate_host_table((5, 16), FP8)

    assert legacy.calls == [
        (((5, 16),), {"dtype": FP8, "device": "cpu", "pin_memory": True})
    ]
    assert tuple(table.shape) == (5, 16)
    assert fake.events == [] and fake.warnings == [] and fake.infos == []


def test_an_empty_host_table_is_not_pinned(fake, legacy) -> None:
    table = exact_pin.allocate_host_table((0, 16), FP8)

    assert legacy.calls == [
        (((0, 16),), {"dtype": FP8, "device": "cpu", "pin_memory": False})
    ]
    assert tuple(table.shape) == (0, 16)
    assert fake.events == []


def test_enabled_exact_path_logs_its_saving_and_skips_the_old_call(
    fake, legacy
) -> None:
    table = exact_pin.allocate_host_table((1000, 48), FP8)

    assert legacy.calls == []
    assert table.data_ptr() % PAGE == 0
    assert len(fake.infos) == 1 and "exact size" in fake.infos[0]
    assert fake.warnings == []


# --- through the real materialize_tables ------------------------------------

ROWS, DIM = 100, 48


def _layer(fake, *, available=None, host_rows=40, budget_rows=None):
    layer_cls, device_calls = pin_boot.layer_class(
        available=available,
        budget=(host_rows if budget_rows is None else budget_rows) * DIM,
        embedding_dim=DIM,
        total_rows=ROWS,
    )
    return layer_cls(), device_calls


def test_materialize_tables_publishes_the_exact_host_table(fake, legacy) -> None:
    layer, device_calls = _layer(fake)

    layer.materialize_tables()

    host = layer.ple_host_storage
    assert tuple(host.shape) == (40, DIM) and host.dtype == FP8
    assert host.data_ptr() % PAGE == 0
    assert host.untyped_storage().nbytes() == 40 * DIM
    assert fake.registered == [(host.data_ptr(), _round_up(40 * DIM), 3)]
    assert legacy.calls == []
    assert tuple(layer.ple_device_table.shape) == (60, DIM)
    assert (layer._device_rows, layer._host_rows) == (60, 40)
    assert layer._device_table_ptr == layer.ple_device_table.data_ptr()
    assert len(device_calls) == 1
    # Idempotent, like before.
    layer.materialize_tables()
    assert layer.ple_host_storage is host and len(fake.registered) == 1


def test_materialize_tables_with_the_switch_off_asks_for_pin_memory(
    fake, legacy, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(exact_pin.EXACT_PIN_ENV, "0")
    layer, _ = _layer(fake)

    layer.materialize_tables()

    assert legacy.calls == [
        (((40, DIM),), {"dtype": FP8, "device": "cpu", "pin_memory": True})
    ]
    assert fake.registered == []
    assert (layer._device_rows, layer._host_rows) == (60, 40)


def test_materialize_tables_without_host_rows_pins_nothing(fake, legacy) -> None:
    layer, _ = _layer(fake, budget_rows=0)

    layer.materialize_tables()

    assert tuple(layer.ple_host_storage.shape) == (0, DIM)
    assert legacy.calls == [
        (((0, DIM),), {"dtype": FP8, "device": "cpu", "pin_memory": False})
    ]
    assert fake.registered == []
    assert (layer._device_rows, layer._host_rows) == (ROWS, 0)


def test_materialize_tables_keeps_the_available_memory_precheck(fake, legacy) -> None:
    layer, _ = _layer(fake, available=40 * DIM - 1)

    with pytest.raises(RuntimeError, match="pinned host memory"):
        layer.materialize_tables()

    assert layer.ple_device_table is None and layer.ple_host_storage is None
    assert fake.registered == [] and legacy.calls == []

    enough, _ = _layer(fake, available=40 * DIM)
    enough.materialize_tables()
    assert enough._host_rows == 40


def test_materialize_tables_registration_failure_still_gets_a_pinned_table(
    fake, legacy
) -> None:
    fake.register_error = RuntimeError(
        "cuMemHostRegister failed: CUDA_ERROR_NOT_SUPPORTED"
    )
    layer, _ = _layer(fake)

    layer.materialize_tables()

    assert len(legacy.calls) == 1 and legacy.calls[0][1]["pin_memory"] is True
    assert tuple(layer.ple_host_storage.shape) == (40, DIM)
    assert len(fake.warnings) == 1


def test_a_host_allocation_failure_leaves_materialization_retryable(
    fake, legacy
) -> None:
    layer, _ = _layer(fake)
    fake.register_error = RuntimeError("registration refused")
    legacy.error = RuntimeError("injected pinned allocation failure")

    with pytest.raises(RuntimeError, match="injected pinned allocation failure"):
        layer.materialize_tables()
    assert layer.ple_device_table is None
    assert layer.ple_host_storage is None
    assert layer._device_table_ptr == 0

    fake.register_error = None
    legacy.error = None
    layer.materialize_tables()
    assert (layer._device_rows, layer._host_rows) == (60, 40)
    assert len(fake.registered) == 1


def test_several_tables_in_one_process_get_separate_mappings(fake) -> None:
    # Each tensor-parallel rank is its own process and builds its own table;
    # in one process that is independent mappings that never overlap.
    tables = [exact_pin.allocate_exact_pinned((300, 48), FP8) for _ in range(4)]
    spans = sorted((t.data_ptr(), t.data_ptr() + _round_up(300 * 48)) for t in tables)
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(spans, spans[1:]))
    assert len(fake.registered) == 4 and fake.unregistered == []
    del tables
    _collect()
    assert len(fake.unregistered) == 4


# --- the probe script's pure parts ------------------------------------------


def _probe():
    import importlib.util

    name = "sx_probe_pinned_memory"
    if name in sys.modules:
        return sys.modules[name]
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "probe_pinned_memory.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GIB = 1024**3
REQUESTED = int(11.92 * GIB)


def _result(consumed: int, checks=None, **extra) -> dict:
    return {
        "requested": REQUESTED,
        "consumed_after_fill": {"MemAvailable": consumed},
        "checks": checks or {},
        **extra,
    }


def test_probe_parses_meminfo_and_status_in_bytes() -> None:
    probe = _probe()
    text = "\n".join(
        [
            "MemTotal:       65839540 kB",
            "MemAvailable:   62914560 kB",
            "Mlocked:               8 kB",
            "VmPin:\t    4096 kB",
            "",
        ]
    )
    wanted = ("MemAvailable", "Mlocked", "Unevictable")
    assert probe.parse_kib_fields(text, wanted) == {
        "MemAvailable": 62914560 * 1024,
        "Mlocked": 8 * 1024,
    }
    assert probe.parse_kib_fields(text, ("VmPin",)) == {"VmPin": 4096 * 1024}


def test_probe_counts_memavailable_as_lost_and_the_rest_as_gained() -> None:
    probe = _probe()
    before = {"meminfo": {"MemAvailable": 100, "Unevictable": 5}, "status": {}}
    after = {"meminfo": {"MemAvailable": 40, "Unevictable": 9}, "status": {}}
    assert probe.consumed(before, after, "MemAvailable", source="meminfo") == 60
    assert probe.consumed(before, after, "Unevictable", source="meminfo") == 4
    assert probe.consumed(before, after, "Mlocked", source="meminfo") is None


def test_probe_passes_the_exact_path_within_two_percent() -> None:
    probe = _probe()
    results = {
        "torch": _result(16 * GIB),
        "exact": _result(int(REQUESTED * 1.004)),
    }
    code, lines = probe.evaluate(results, 0.02)
    assert code == 0
    assert any(line.startswith("PASS [exact]") for line in lines)
    assert any(
        "16.000 GiB" in line and line.startswith("INFO [torch]") for line in lines
    )
    assert any(line.startswith("INFO saving") for line in lines)


def test_probe_fails_the_exact_path_above_two_percent() -> None:
    probe = _probe()
    code, lines = probe.evaluate({"exact": _result(int(REQUESTED * 1.0201))}, 0.02)
    assert code == 1 and any(line.startswith("FAIL [exact]") for line in lines)
    code, _ = probe.evaluate({"exact": _result(16 * GIB)}, 0.02)
    assert code == 1
    code, _ = probe.evaluate({"exact": _result(16 * GIB)}, 0.5)
    assert code == 0


def test_probe_reports_a_failed_usability_check_before_the_overhead() -> None:
    probe = _probe()
    checks = {"uva_view_made_no_copy": {"ok": False, "detail": "copied"}}
    code, lines = probe.evaluate({"exact": _result(16 * GIB, checks)}, 0.02)
    assert code == 2
    assert any("uva_view_made_no_copy: copied" in line for line in lines)
    ok = {"is_pinned": {"ok": True, "detail": ""}}
    assert probe.evaluate({"torch": _result(16 * GIB, ok)}, 0.02)[0] == 0


def test_probe_reports_an_allocation_failure_of_the_exact_path() -> None:
    probe = _probe()
    failed = {
        "mode": "exact",
        "requested": REQUESTED,
        "checks": {"allocate": {"ok": False, "detail": "RuntimeError: refused"}},
    }
    code, lines = probe.evaluate({"exact": failed}, 0.02)
    assert code == 2 and any("allocate: RuntimeError: refused" in x for x in lines)


def test_probe_environment_problems_exit_with_three() -> None:
    probe = _probe()
    broken = {"mode": "exact", "environment_error": "no CUDA"}
    code, lines = probe.evaluate({"torch": broken, "exact": broken}, 0.02)
    assert code == 3 and any("environment" in line for line in lines)


@pytest.mark.skipif(torch.cuda.is_available(), reason="needs a machine without CUDA")
def test_probe_without_cuda_exits_with_three_and_says_why(capfd) -> None:
    probe = _probe()
    assert probe.main(["--mode", "exact", "--gib", "0.01"]) == 3
    out = capfd.readouterr().out
    assert "torch.cuda.is_available() is False" in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", *sys.argv[1:]]))
