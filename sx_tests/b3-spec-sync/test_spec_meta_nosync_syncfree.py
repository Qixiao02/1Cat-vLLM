# SPDX-License-Identifier: Apache-2.0
"""SX_OPT_SPEC_META_NOSYNC: the GDN / PLE short-conv spec builders never
block the host (1 GPU, cuda:0).

Two independent checks:

1. torch sync-debug: every new-path build runs under
   ``torch.cuda.set_sync_debug_mode("error")``, which raises on any torch
   level host sync (``.item()`` of a CUDA tensor, ``nonzero`` from boolean
   indexing, blocking ``.to(device)``, stream/device synchronize). The legacy
   builders are the control: they must raise.

2. Busy GPU: a ~0.1 s spin kernel (``torch.cuda._sleep``) is queued, the
   metadata is built, and the event recorded after the spin kernel must still
   be pending when the build returns, i.e. the host never waited for the
   stream. This also catches driver-level waits that sync-debug cannot see
   (pageable host->device copies). The legacy builders are the control: the
   event must be complete after their build.
   Mixed steps that contain prefill chunks also run the shared
   ``compute_causal_conv1d_metadata`` helper (backends/utils.py, not part of
   this change, used unchanged by the no-MTP lane), whose small H2D copies are
   pageable; that case is reported (xfail) instead of asserted.

  /opt/venv/bin/python -m pytest -q sx_tests/b3-spec-sync/test_spec_meta_nosync_syncfree.py
"""

from __future__ import annotations

import contextlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _spec_meta_common import (  # noqa: E402
    B_VALUES,
    K_VALUES,
    batches_for,
    build_gdn,
    build_ple,
    make_gdn_builder,
    make_inputs,
    make_ple_builder,
    set_gdn_envs,
)

import torch  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs 1 CUDA GPU"
)

DEVICE = "cuda"
SLEEP_CYCLES = int(1.5e8)  # ~0.1 s at V100 clocks (a build takes a few ms)


@contextlib.contextmanager
def sync_debug_error():
    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(previous)


def _builders(monkeypatch, k, mode, *, nosync):
    return {
        full: (
            make_gdn_builder(monkeypatch, k, mode, DEVICE, full=full, nosync=nosync),
            make_ple_builder(k, mode, DEVICE, full=full, nosync=nosync),
        )
        for full in (True, False)
    }


def _build_both(builders, inputs, *, current_state=False):
    gdn, ple = builders[inputs.batch.full]
    return (
        build_gdn(gdn, inputs, current_state=current_state),
        build_ple(ple, inputs),
    )


def _spec_batches(k, b, seed):
    # Steps without any spec row run the untouched legacy branch of the spec
    # builders (and the shared no-MTP helpers); they are out of scope here.
    return [
        batch
        for batch in batches_for(k, b, seed=seed)
        if not batch.name.startswith("nospec")
    ]


@pytest.mark.parametrize("mode", ("align", "none"))
@pytest.mark.parametrize("k", K_VALUES)
@pytest.mark.parametrize("legacy_routing", (True, False))
def test_new_builders_pass_sync_debug_error(monkeypatch, mode, k, legacy_routing):
    set_gdn_envs(monkeypatch, legacy_routing=legacy_routing, legacy_slot0=True)
    builders = _builders(monkeypatch, k, mode, nosync=True)
    for b in B_VALUES:
        for index, batch in enumerate(_spec_batches(k, b, seed=31 * k + b)):
            inputs = make_inputs(batch, mode, DEVICE, seed=index)
            _build_both(builders, inputs)  # warm-up (allocator pools)
            _build_both(builders, inputs, current_state=True)
            torch.cuda.synchronize()
            try:
                with sync_debug_error():
                    _build_both(builders, inputs)
                    # Also with explicit slot selectors (GDN contract path) and
                    # the V1-runner align-mode current_state_block_ids path.
                    build_gdn(builders[batch.full][0], inputs, selectors=True)
                    build_gdn(
                        builders[batch.full][0],
                        inputs,
                        selectors=True,
                        current_state=True,
                    )
            except RuntimeError as exc:  # pragma: no cover - failure report
                pytest.fail(f"{batch.name} mode={mode}: host sync: {exc}")
            torch.cuda.synchronize()


@pytest.mark.parametrize("k", K_VALUES)
def test_legacy_builders_fail_sync_debug(monkeypatch, k):
    """Control: the harness detects the syncs the change removes."""
    set_gdn_envs(monkeypatch, legacy_routing=True, legacy_slot0=True)
    builders = _builders(monkeypatch, k, "align", nosync=False)
    for batch in batches_for(k, 4, seed=k)[:2]:  # pure verify full / eager
        inputs = make_inputs(batch, "align", DEVICE, seed=0)
        gdn, ple = builders[batch.full]
        build_gdn(gdn, inputs)
        build_ple(ple, inputs)
        torch.cuda.synchronize()
        with sync_debug_error():
            with pytest.raises(RuntimeError):
                build_gdn(gdn, inputs)
            with pytest.raises(RuntimeError):
                build_ple(ple, inputs)
        torch.cuda.synchronize()


def _host_waited_once(fn) -> bool:
    torch.cuda.synchronize()
    done = torch.cuda.Event()
    torch.cuda._sleep(SLEEP_CYCLES)
    done.record()
    fn()
    waited = done.query()
    torch.cuda.synchronize()
    return waited


def _host_waited(fn, attempts: int = 3) -> bool:
    """True if the host waited for the busy GPU on every attempt.

    Pinned staging blocks freed behind a busy stream are recycled only once
    their copy event completes, so the first overlapped builds may still grow
    the caching host allocator (cudaHostAlloc). Steady state is what counts:
    one run-ahead build out of ``attempts`` proves the path has no sync.
    """
    return all(_host_waited_once(fn) for _ in range(attempts))


@pytest.mark.parametrize("mode", ("align", "none"))
@pytest.mark.parametrize("k", K_VALUES)
def test_host_runs_ahead_of_busy_gpu(monkeypatch, mode, k):
    set_gdn_envs(monkeypatch, legacy_routing=True, legacy_slot0=True)
    new = _builders(monkeypatch, k, mode, nosync=True)
    blocked: list[str] = []
    prefill_blocked: list[str] = []
    for b in (1, 4, 12, 24):
        for index, batch in enumerate(_spec_batches(k, b, seed=77 * k + b)):
            inputs = make_inputs(batch, mode, DEVICE, seed=index)
            _build_both(new, inputs)  # warm-up
            if _host_waited(lambda: _build_both(new, inputs)):
                (prefill_blocked if batch.has_prefill else blocked).append(
                    batch.name
                )
    assert not blocked, f"host waited for the GPU in: {blocked}"
    if prefill_blocked:
        pytest.xfail(
            "host waited on mixed steps with prefill chunks (pageable copies "
            f"in backends/utils.compute_causal_conv1d_metadata): {prefill_blocked}"
        )


@pytest.mark.parametrize("k", K_VALUES)
def test_legacy_builders_wait_for_busy_gpu(monkeypatch, k):
    """Control: the busy-GPU probe sees the legacy host syncs."""
    set_gdn_envs(monkeypatch, legacy_routing=True, legacy_slot0=True)
    old = _builders(monkeypatch, k, "align", nosync=False)
    batch = batches_for(k, 8, seed=k)[0]  # pure verify, FULL graph
    inputs = make_inputs(batch, "align", DEVICE, seed=0)
    gdn, ple = old[batch.full]
    build_gdn(gdn, inputs)
    build_ple(ple, inputs)
    assert _host_waited(lambda: build_gdn(gdn, inputs))
    assert _host_waited(lambda: build_ple(ple, inputs))
