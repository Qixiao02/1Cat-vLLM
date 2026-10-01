# SPDX-License-Identifier: Apache-2.0
"""Old vs new cost per tensor name of the FusedMoE.load_weights expert lookup.

    python sx_tests/load-index/bench_moe_load_index.py [--names 150000] [--old-names 4000]

CPU only, no vLLM install needed. Prints, per tensor name (microseconds):

* lookup: the selection alone (the original loop over 1536 entries against the
  index lookup), for names that match an expert entry and for names that match
  nothing (those pay the scan again as the fallback);
* generator: the whole FusedMoE.load_weights generator with a no-op
  weight_loader, switch off against on, on 1-D tensors the size of a scale.

The original loop is slow (about 100 us per name), so it is timed on the first
``--old-names`` names and the index on all of them; per-name figures are
comparable, totals are extrapolated and labelled as such.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import load_boot  # noqa: E402
import reference_original  # noqa: E402

idx = load_boot.expert_mapping_index()
make_mapping = load_boot.make_expert_params_mapping()
load_weights_new = load_boot.fused_moe_load_weights()
load_weights_old = reference_original.load_weights_original


class _Model:
    def named_parameters(self):
        return []


class FakeMoE:
    def __init__(self, layer_name, mapping):
        self.layer_name = layer_name
        self.expert_mapping = mapping

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return None

    def weight_loader(self, **kwargs):
        return True


def names(count: int, num_experts: int = 512, layers: int = 48):
    """Expert tensor names in checkpoint order: layer, expert, projection, tensor."""
    out = []
    suffixes = ("weight", "weight_scale", "weight_scale_2", "input_scale")
    for layer in range(layers):
        for expert in range(num_experts):
            for proj in ("down_proj", "gate_proj", "up_proj"):
                for suffix in suffixes:
                    out.append((f"model.layers.{layer}.mlp.experts", f"{expert}.{proj}.{suffix}"))
                    if len(out) == count:
                        return out
    return out


def timed(fn, repeat: int = 3):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", type=int, default=150_000)
    parser.add_argument("--old-names", type=int, default=4000)
    args = parser.parse_args()

    mapping = make_mapping(_Model(), "gate_proj", "down_proj", "up_proj", 512, 0, True)
    index = idx.build_expert_mapping_index(mapping)
    start = time.perf_counter()
    for _ in range(20):
        idx.build_expert_mapping_index(mapping)
    build_ms = (time.perf_counter() - start) / 20 * 1e3
    print(f"mapping: {len(mapping)} entries; index build {build_ms:.2f} ms per layer")

    pairs = names(args.names)
    quals = [f"{layer}.{expert_name}" for layer, expert_name in pairs]
    quals_none = [f"model.layers.3.mlp.shared_expert.{i}.gate_proj.weight" for i in range(len(quals))]
    n_old = min(args.old_names, len(quals))

    def old_select(qs):
        out = 0
        for q in qs:
            for param_name, weight_name, expert_id, shard_id in mapping:
                if weight_name not in q:
                    continue
                out += 1
        return out

    def new_select(qs):
        out = 0
        for q in qs:
            for param_name, weight_name, expert_id, shard_id in index.matches(q, mapping):
                if weight_name not in q:
                    continue
                out += 1
        return out

    rows = []
    for label, qs in (("names that match an expert entry", quals),
                      ("names that match no entry (fallback scan)", quals_none)):
        t_old, n1 = timed(lambda: old_select(qs[:n_old]), 2)
        t_new, n2 = timed(lambda: new_select(qs[:n_old]), 3)
        assert n1 == n2
        t_new_all, _ = timed(lambda: new_select(qs), 1)
        rows.append((label, t_old / n_old * 1e6, t_new_all / len(qs) * 1e6))
    print(f"\nlookup (selection only), us per name  [old: first {n_old} names; new: {len(quals)}]")
    for label, old_us, new_us in rows:
        print(f"  {label:45s} old {old_us:8.2f}   new {new_us:7.2f}   x{old_us / new_us:7.1f}")
    old_us, new_us = rows[0][1], rows[0][2]
    how = "measured" if n_old >= len(quals) else "extrapolated"
    print(f"  total for {len(quals)} names: old {old_us * len(quals) / 1e6:.1f} s ({how}), "
          f"new {new_us * len(quals) / 1e6:.2f} s")

    # The whole generator, no-op weight_loader, tiny tensors.
    tensors = [torch.zeros(1) for _ in range(256)]
    weights = [(en, tensors[i % 256]) for i, (_, en) in enumerate(pairs)]
    moe = FakeMoE(pairs[0][0], mapping)

    def run(fn, count):
        n = 0
        for _ in fn(moe, iter(weights[:count])):
            n += 1
        return n

    os.environ["SX_OPT_MOE_LOAD_INDEX"] = "0"
    t_off, n_off = timed(lambda: run(load_weights_new, n_old), 2)
    os.environ["SX_OPT_MOE_LOAD_INDEX"] = "1"
    t_on, n_on = timed(lambda: run(load_weights_new, n_old), 3)
    assert n_off == n_on
    t_on_all, _ = timed(lambda: run(load_weights_new, len(weights)), 1)
    off_us, on_us = t_off / n_old * 1e6, t_on_all / len(weights) * 1e6
    print(f"\ngenerator (no-op weight_loader), us per tensor  [off: first {n_old}; on: {len(weights)}]")
    print(f"  switch off {off_us:8.2f}   switch on {on_us:7.2f}   x{off_us / on_us:6.1f}")
    how = "measured" if n_old >= len(weights) else "extrapolated"
    print(f"  total for {len(weights)} tensors: off {off_us * len(weights) / 1e6:.1f} s "
          f"({how}), on {on_us * len(weights) / 1e6:.2f} s")


if __name__ == "__main__":
    main()
