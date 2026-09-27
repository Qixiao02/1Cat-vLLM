# SPDX-License-Identifier: Apache-2.0
"""SX_OPT_SPEC_META_NOSYNC: new GDN / PLE short-conv spec metadata == old.

For every batch the legacy builder (``_sx_spec_meta_nosync = False``, i.e.
SX_OPT_SPEC_META_NOSYNC=0) and the sync-free builder build metadata from the
same inputs; every dataclass field must match: ints exactly, tensors in
dtype, device, shape, layout (contiguity + strides of non-singleton dims)
and values (torch.equal). ``nums_dict`` is compared recursively.

Grid: k (num_speculative_tokens) 1..4, B live requests {1,2,4,8,12,16,24}
(verify widths B*(k+1)), mamba cache modes align (production) / none / all,
batch kinds
  * verify_full   pure MTP verify, FULL-graph builder, padded to the MTP
                  request sizes {1,2,3,4,6,8,12,16,24} with zero-length rows,
  * verify_eager  pure verify through a non-FULL builder (e.g. the k4 x 24
                  verify that has no FULL graph),
  * verify_eager_ragged  pure verify with per-request draft counts 0..k
                  (adaptive k / requests near max_tokens; 0 drafts = V1-runner
                  spec row with a 1-token query),
  * nospec_*      steps of the spec builders without any spec row (plain
                  decodes, decodes + prefills): legacy branch on both paths,
  * mixed_*       spec decodes + plain decodes (0 drafts) + prefill chunks
                  (2..7840 tokens), spec rows first / shuffled / last,
                  with and without zero-length tail rows, without prefills,
                  and with ragged draft counts,
the GDN mixed routing / non-spec slot-0 legacy switches in all four
combinations, and (GDN) explicit spec_state_slot_selectors. Also: the
V1-runner align-mode current_state_block_ids path, capture builds
(build_for_cudagraph_capture), repeated builds on one builder with
alternating widths (persistent FULL buffers), FULL-capable builders whose
verify width exceeds max_cudagraph_capture_size, the opt-in debug fences
(VLLM_SM70_GDN_STATE_CONTRACT_ASSERT=1), and a pure batch with padding ahead
of spec rows (GDN rejects it on both paths).

Every tensor that was 16-byte aligned on the legacy path must stay aligned
(Triton/TileLang kernels specialize on or assume pointer alignment), which
is why mixed batches never hand out offset views of per-row tensors. A build
that raises must raise the same exception type on both paths (the only such
case is a pre-existing limitation of the untouched no-spec PLE branch in
mamba_cache_mode "all").

Runs on CPU and, when present, on 1 GPU (cuda:0):

  /opt/venv/bin/python -m pytest -q sx_tests/b3-spec-sync/test_spec_meta_nosync_equal.py
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _spec_meta_common import (  # noqa: E402
    B_VALUES,
    K_VALUES,
    MODES,
    batches_for,
    build_gdn,
    build_outcome,
    build_ple,
    devices,
    fake_vllm_config,
    make_gdn_builder,
    make_inputs,
    make_ple_builder,
    metadata_diff,
    outcome_diff,
    set_gdn_envs,
    snapshot,
)

ENV_COMBOS = (
    (True, True),  # production defaults
    (True, False),
    (False, True),
    (False, False),
)


def _gdn_builders(monkeypatch, k, mode, device):
    return {
        (full, nosync): make_gdn_builder(
            monkeypatch, k, mode, device, full=full, nosync=nosync
        )
        for full in (True, False)
        for nosync in (False, True)
    }


def _ple_builders(k, mode, device):
    return {
        (full, nosync): make_ple_builder(k, mode, device, full=full, nosync=nosync)
        for full in (True, False)
        for nosync in (False, True)
    }


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("b", B_VALUES)
@pytest.mark.parametrize("k", K_VALUES)
def test_gdn_spec_metadata_equal(monkeypatch, device, mode, b, k):
    builders = _gdn_builders(monkeypatch, k, mode, device)
    failures: list[str] = []
    checked = 0
    for index, batch in enumerate(batches_for(k, b, seed=1000 * k + b)):
        inputs = make_inputs(batch, mode, device, seed=index)
        combos = ENV_COMBOS if batch.name.startswith("mixed") else ENV_COMBOS[:2]
        for legacy_routing, legacy_slot0 in combos:
            set_gdn_envs(
                monkeypatch,
                legacy_routing=legacy_routing,
                legacy_slot0=legacy_slot0,
            )
            for selectors in (False, True):
                old_b = builders[(batch.full, False)]
                new_b = builders[(batch.full, True)]
                old = build_outcome(build_gdn, old_b, inputs, selectors=selectors)
                new = build_outcome(build_gdn, new_b, inputs, selectors=selectors)
                diff = outcome_diff(old, new)
                checked += 1
                if diff:
                    failures.append(
                        f"{batch.name} routing={legacy_routing} slot0={legacy_slot0} "
                        f"selectors={selectors}:\n    " + "\n    ".join(diff)
                    )
    assert checked > 0
    assert not failures, "\n".join(failures[:10])


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("b", B_VALUES)
@pytest.mark.parametrize("k", K_VALUES)
def test_ple_short_conv_spec_metadata_equal(device, mode, b, k):
    builders = _ple_builders(k, mode, device)
    failures: list[str] = []
    for index, batch in enumerate(batches_for(k, b, seed=2000 * k + b)):
        inputs = make_inputs(batch, mode, device, seed=index)
        old = build_outcome(build_ple, builders[(batch.full, False)], inputs)
        new = build_outcome(build_ple, builders[(batch.full, True)], inputs)
        diff = outcome_diff(old, new)
        if diff:
            failures.append(f"{batch.name}:\n    " + "\n    ".join(diff))
    assert not failures, "\n".join(failures[:10])


@pytest.mark.parametrize("device", devices())
def test_repeated_builds_stay_equal(monkeypatch, device):
    """Persistent FULL-graph buffers: alternating widths on the same builder
    (grow/shrink the padded tail) keep matching the legacy builder."""
    k, mode = 4, "align"
    gdn_old = make_gdn_builder(monkeypatch, k, mode, device, full=True, nosync=False)
    gdn_new = make_gdn_builder(monkeypatch, k, mode, device, full=True, nosync=True)
    ple_old = make_ple_builder(k, mode, device, full=True, nosync=False)
    ple_new = make_ple_builder(k, mode, device, full=True, nosync=True)
    set_gdn_envs(monkeypatch, legacy_routing=True, legacy_slot0=True)
    failures = []
    for step, b in enumerate((24, 1, 16, 3, 8, 24, 2, 12, 5, 1)):
        for batch in batches_for(k, b, seed=step)[:1]:
            inputs = make_inputs(batch, mode, device, seed=step)
            for name, old_b, new_b, fn in (
                ("gdn", gdn_old, gdn_new, build_gdn),
                ("ple", ple_old, ple_new, build_ple),
            ):
                diff = metadata_diff(
                    snapshot(fn(old_b, inputs)), snapshot(fn(new_b, inputs))
                )
                if diff:
                    failures.append(f"step{step} {name} {batch.name}: {diff[:3]}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("capture", (16, 64))
def test_full_builder_oversize_verify_equal(monkeypatch, device, capture):
    """FULL-capable builders whose verify width exceeds the capture size
    (e.g. k4 x 24 = 120 tokens) take the eager metadata route."""
    set_gdn_envs(monkeypatch, legacy_routing=True, legacy_slot0=True)
    failures = []
    for k in (2, 4):
        mode = "align"
        cfg = fake_vllm_config(k, mode, full=True, capture=capture)
        gdn_old = make_gdn_builder(monkeypatch, k, mode, device, full=True,
                                   nosync=False, vllm_config=cfg)
        gdn_new = make_gdn_builder(monkeypatch, k, mode, device, full=True,
                                   nosync=True, vllm_config=cfg)
        ple_old = make_ple_builder(k, mode, device, full=True, nosync=False,
                                   vllm_config=cfg)
        ple_new = make_ple_builder(k, mode, device, full=True, nosync=True,
                                   vllm_config=cfg)
        for b in (2, 8, 12, 16, 24):
            for index, batch in enumerate(batches_for(k, b, seed=b)[:2]):
                inputs = make_inputs(batch, mode, device, seed=index)
                for name, old_b, new_b, fn in (
                    ("gdn", gdn_old, gdn_new, build_gdn),
                    ("ple", ple_old, ple_new, build_ple),
                ):
                    diff = metadata_diff(
                        snapshot(fn(old_b, inputs)), snapshot(fn(new_b, inputs))
                    )
                    if diff:
                        failures.append(f"k{k} {name} {batch.name}: {diff[:3]}")
    assert not failures, "; ".join(failures)


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("b", B_VALUES)
@pytest.mark.parametrize("k", K_VALUES)
def test_gdn_current_state_block_ids_equal(monkeypatch, device, b, k):
    """V1 GPUModelRunner align mode passes ``current_state_block_ids`` (its
    ``current_mamba_state_block_ids`` buffer, width 1 + k, PAD_SLOT_ID for
    rows without a state block) and the runner's slot-selector buffer."""
    mode = "align"
    builders = _gdn_builders(monkeypatch, k, mode, device)
    failures: list[str] = []
    for index, batch in enumerate(batches_for(k, b, seed=3000 * k + b)):
        inputs = make_inputs(batch, mode, device, seed=index)
        for legacy_routing, legacy_slot0 in ENV_COMBOS:
            set_gdn_envs(
                monkeypatch,
                legacy_routing=legacy_routing,
                legacy_slot0=legacy_slot0,
            )
            for selectors in (False, True):
                old, new = (
                    build_outcome(
                        build_gdn,
                        builders[(batch.full, nosync)],
                        inputs,
                        selectors=selectors,
                        current_state=True,
                    )
                    for nosync in (False, True)
                )
                diff = outcome_diff(old, new)
                if diff:
                    failures.append(
                        f"{batch.name} routing={legacy_routing} slot0={legacy_slot0} "
                        f"selectors={selectors}:\n    " + "\n    ".join(diff)
                    )
    assert not failures, "\n".join(failures[:10])


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("k", K_VALUES)
def test_capture_builds_equal(monkeypatch, device, k):
    """build_for_cudagraph_capture (FULL verify graphs, padded request sizes):
    the persistent buffers the graphs are captured on get the same content."""
    set_gdn_envs(monkeypatch, legacy_routing=True, legacy_slot0=True)
    mode = "align"
    gdn_old = make_gdn_builder(monkeypatch, k, mode, device, full=True, nosync=False)
    gdn_new = make_gdn_builder(monkeypatch, k, mode, device, full=True, nosync=True)
    ple_old = make_ple_builder(k, mode, device, full=True, nosync=False)
    ple_new = make_ple_builder(k, mode, device, full=True, nosync=True)
    failures = []
    for b in (1, 2, 3, 4, 5, 6, 8, 12, 16, 24):
        batch = batches_for(k, b, seed=b)[0]  # pure verify, FULL-graph padding
        inputs = make_inputs(batch, mode, device, seed=b)
        for name, old_b, new_b in (
            ("gdn", gdn_old, gdn_new),
            ("ple", ple_old, ple_new),
        ):
            diff = metadata_diff(
                snapshot(old_b.build_for_cudagraph_capture(inputs.common())),
                snapshot(new_b.build_for_cudagraph_capture(inputs.common())),
            )
            if diff:
                failures.append(f"{name} {batch.name}: {diff[:3]}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize("k", (1, 4))
def test_pure_batch_with_leading_padding_behaves_the_same(monkeypatch, device, k):
    """A pure verify batch whose zero-length rows precede spec rows breaks the
    "padding at the back" contract. GDN rejects it on both paths (the host
    check keeps the old device assert's verdict); PLE stays equal (it takes
    the non-prefix index path)."""
    from _spec_meta_common import SpecBatch

    set_gdn_envs(monkeypatch, legacy_routing=True, legacy_slot0=True)
    mode = "align"
    batch = SpecBatch(
        name=f"pure_lead_pad_k{k}",
        k=k,
        query_lens=[0, k + 1, 0, k + 1, k + 1],
        drafts=[-1, k, -1, k, k],
        seq_lens=[0, 900 + k, 0, 5000 + k, 70 + k],
        accepted=[1, 1, 1, k + 1, 2],
        num_actual_tokens=3 * (k + 1),
        full=False,
    )
    inputs = make_inputs(batch, mode, device, seed=7)
    outcomes = []
    for nosync in (False, True):
        builder = make_gdn_builder(
            monkeypatch, k, mode, device, full=False, nosync=nosync
        )
        try:
            build_gdn(builder, inputs)
            outcomes.append("ok")
        except AssertionError:
            outcomes.append("assert")
    assert outcomes[0] == outcomes[1], outcomes
    ple_old = make_ple_builder(k, mode, device, full=False, nosync=False)
    ple_new = make_ple_builder(k, mode, device, full=False, nosync=True)
    diff = metadata_diff(
        snapshot(build_ple(ple_old, inputs)), snapshot(build_ple(ple_new, inputs))
    )
    assert not diff, diff


@pytest.mark.parametrize("device", devices())
def test_debug_contract_assert_env_equal(monkeypatch, device):
    """VLLM_SM70_GDN_STATE_CONTRACT_ASSERT=1 (opt-in, syncing debug fences):
    the new path also re-runs the legacy device-side query_start_loc check;
    valid batches build equal metadata on both paths."""
    monkeypatch.setenv("VLLM_SM70_GDN_STATE_CONTRACT_ASSERT", "1")
    set_gdn_envs(monkeypatch, legacy_routing=True, legacy_slot0=True)
    failures = []
    for k in (1, 4):
        builders = _gdn_builders(monkeypatch, k, "align", device)
        for b in (1, 3, 8, 24):
            for index, batch in enumerate(batches_for(k, b, seed=b)):
                inputs = make_inputs(batch, "align", device, seed=index)
                for current_state in (False, True):
                    old, new = (
                        build_outcome(
                            build_gdn,
                            builders[(batch.full, nosync)],
                            inputs,
                            selectors=True,
                            current_state=current_state,
                        )
                        for nosync in (False, True)
                    )
                    assert old[0] == "ok", (batch.name, old)
                    diff = outcome_diff(old, new)
                    if diff:
                        failures.append(f"k{k} {batch.name}: {diff[:3]}")
    assert not failures, "\n".join(failures)
