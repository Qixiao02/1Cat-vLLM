# SPDX-License-Identifier: Apache-2.0
"""Cost per checkpoint tensor of the Python loading stack, switch by switch.

    python sx_tests/load-index/bench_load_pipeline.py [--layers 3] [--experts 512]

CPU only, no vLLM install needed (see ``pipeline_harness.py``): three nested
AutoWeightsLoader levels, the QSA scale remap and FusedMoE.load_weights from the
real source, with a toy weight_loader that does nothing, so the figures are the
Python plumbing around the copy into the model and nothing else (no disk, no
GPU). Microseconds per tensor, best of 3.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import load_boot  # noqa: E402
import pipeline_harness as ph  # noqa: E402

U = ph.utils_namespace()
M = ph.model_namespace()
make_mapping = load_boot.make_expert_params_mapping()


class _Model:
    def named_parameters(self):
        return []


def per_tensor_us(layers: int, experts: int, qsa_ids, moe_index: bool,
                  can_skip: bool, qsa_remap: bool, repeat: int = 3) -> float:
    os.environ["SX_OPT_MOE_LOAD_INDEX"] = "1" if moe_index else "0"
    U["_SX_LOAD_CAN_SKIP"] = can_skip
    M["_SX_LOAD_QSA_REMAP"] = qsa_remap
    best = float("inf")
    for _ in range(repeat):
        model, counter = ph.build_model(
            layers,
            lambda: make_mapping(_Model(), "gate_proj", "down_proj", "up_proj",
                                 experts, 0, True),
            qsa_ids=qsa_ids,
        )
        counter.record = False
        names = list(ph.checkpoint_names(layers, experts, qsa_ids=qsa_ids))
        start = time.perf_counter()
        model.load_weights(ph.tensor_stream(names))
        best = min(best, time.perf_counter() - start)
    return best / len(names) * 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--experts", type=int, default=512)
    args = parser.parse_args()
    qsa_ids = tuple(range(1, 48, 4))  # 12 QSA layers, as Flash-Next
    rows = [
        ("all switches off (original code paths)", (False, False, False)),
        ("+ SX_OPT_MOE_LOAD_INDEX", (True, False, False)),
        ("+ SX_OPT_LOAD_CAN_SKIP", (True, True, False)),
        ("+ SX_OPT_LOAD_QSA_REMAP", (True, False, True)),
        ("all three on (default)", (True, True, True)),
    ]
    print(f"{args.layers} layers x {args.experts} experts, 12 QSA layer ids; "
          "us per checkpoint tensor")
    base = None
    for label, flags in rows:
        us = per_tensor_us(args.layers, args.experts, qsa_ids, *flags)
        base = base or us
        print(f"  {label:42s} {us:7.2f}   x{base / us:5.2f}")


if __name__ == "__main__":
    main()
