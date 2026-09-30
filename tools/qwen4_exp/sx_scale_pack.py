#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Make a QSA E4M3 scale pack from a calibration of our own checkpoint.

materialize_qsa_scale_overlay.py takes the target K/V scales from a scale
pack: model-kvscales.safetensors plus a kvscales-manifest.json that binds it
to one base checkpoint. Published packs exist for the published checkpoints
only. For any other checkpoint the pack comes from a calibration report that
was collected with MTP on (the target QSA layers plus the MTP layer):

    sx_scale_pack.py target-report      the report without the MTP layer
    qsa_kv_calibration.py overlay       model-kvscales.safetensors from it
    sx_scale_pack.py manifest           the manifest for that directory

The same full report then gives the materializer the two MTP scales
(--mtp-report).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import regex as re

INDEX_FILENAME = "model.safetensors.index.json"
MANIFEST_FILENAME = "kvscales-manifest.json"
SCALE_FILENAME = "model-kvscales.safetensors"
_SCALE_NAME = re.compile(r"layers\.(\d+)\.self_attn\.([kv])_scale$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scale_key(name: str) -> tuple[int, str]:
    match = _SCALE_NAME.search(name)
    if match is None:
        raise ValueError(f"Unrecognized QSA scale tensor name: {name}")
    return int(match.group(1)), match.group(2)


def _shard_tensor_names(path: Path) -> list[str]:
    with path.open("rb") as handle:
        header = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
    return sorted(name for name in header if name != "__metadata__")


def target_report(args: argparse.Namespace) -> None:
    """Drop the MTP layer: the overlay subcommand maps every tensor of its
    report onto the target model."""
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Target report already exists: {output}")
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    tensors = report["tensors"]
    target = {
        name: details
        for name, details in tensors.items()
        if _scale_key(name)[0] != args.mtp_layer_id
    }
    if len(tensors) - len(target) != 2:
        raise ValueError(
            f"Expected K and V of MTP layer {args.mtp_layer_id} in the report, "
            f"found {len(tensors) - len(target)} tensors"
        )
    report["tensors"] = target
    report["tensor_count"] = len(target)
    report["qsa_layer_ids"] = sorted({_scale_key(name)[0] for name in target})
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(target)} target scales to {output}")


def manifest(args: argparse.Namespace) -> None:
    pack = Path(args.pack_dir)
    path = pack / MANIFEST_FILENAME
    if path.exists():
        raise FileExistsError(f"Manifest already exists: {path}")
    tensors = json.loads(Path(args.report).read_text(encoding="utf-8"))["tensors"]
    scale_file = pack / SCALE_FILENAME
    names = _shard_tensor_names(scale_file)
    if sorted(map(_scale_key, names)) != sorted(map(_scale_key, tensors)):
        raise ValueError("Scale shard and target report cover different tensors")
    document = {
        "schema_version": 1,
        "artifact_id": args.artifact_id,
        "scale_contract": "e4m3fn(x/scale), max_abs/448",
        "base_checkpoint": {
            "path": str(Path(args.base_checkpoint).resolve()),
            "index_sha256": _sha256(Path(args.base_checkpoint) / INDEX_FILENAME),
        },
        "scale_file": {
            "filename": SCALE_FILENAME,
            "sha256": _sha256(scale_file),
            "tensor_names": names,
            "tensor_count": len(names),
        },
        # The materializer's envelope source compares these maxima with a
        # later report.
        "tensors": [
            {
                "layer": _scale_key(name)[0],
                "kind": _scale_key(name)[1],
                "max_abs": float(details["max_abs"]),
                "scale": float(details["scale"]),
            }
            for name, details in sorted(tensors.items())
        ],
    }
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote {path} for {len(names)} target scales")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    target = subparsers.add_parser("target-report")
    target.add_argument("--report", required=True)
    target.add_argument("--output", required=True)
    target.add_argument("--mtp-layer-id", type=int, default=48)
    target.set_defaults(func=target_report)

    pack = subparsers.add_parser("manifest")
    pack.add_argument("--pack-dir", required=True)
    pack.add_argument("--base-checkpoint", required=True)
    pack.add_argument("--report", required=True)
    pack.add_argument("--artifact-id", required=True)
    pack.set_defaults(func=manifest)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
