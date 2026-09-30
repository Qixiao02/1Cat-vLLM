# SPDX-License-Identifier: Apache-2.0
"""Scale overlay for a checkpoint of our own, from traces to the served view.

Run (CPU, no vLLM install and no GPU needed):

    python -m pytest -q sx_tests/e4m3-mtp/test_scale_pack.py

Runs the offline part of the calibration procedure on a synthetic base
checkpoint and synthetic observer traces of 12 target QSA layers plus the MTP
layer (id 48):

    qsa_kv_calibration.py summarize --expected-layers 13
    sx_scale_pack.py target-report
    qsa_kv_calibration.py overlay        (needs torch + safetensors; without
                                          safetensors its scale shard is
                                          written by the materializer's own
                                          writer, same layout)
    sx_scale_pack.py manifest
    materialize_qsa_scale_overlay.py --mtp-report

Asserted: every scale is max_abs / 448 rounded up to FP32 (target) or as
reported (MTP), the 24 + 2 names the engine loads are in the merged index
under the two shard files, and the pack is refused for a checkpoint with a
different index and for a report that does not match the scale shard.
Expected: all pass. The base directory holds nothing but its index, so the
materializer has no entry to link and this also runs where symlinks cannot
be created.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e4m3_boot  # noqa: E402

e4m3_boot.provide_regex()
TOOLS = Path(e4m3_boot.REPO) / "tools" / "qwen4_exp"
QSA_LAYERS = list(range(3, 48, 4))
MTP_LAYER = 48
E4M3_MAX = 448.0


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(f"sx_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


calibration = _tool("qsa_kv_calibration")
materializer = _tool("materialize_qsa_scale_overlay")
scale_pack = _tool("sx_scale_pack")


def _max_abs(layer: int, kind: str) -> float:
    return (6.0 if kind == "k" else 20.0) + 0.37 * layer


def _stat(max_abs: float) -> dict:
    histogram = [0] * 4096
    histogram[2400] = 100
    return {
        "count": 100,
        "finite_count": 100,
        "nonzero_count": 100,
        "max_abs": max_abs,
        "histogram": histogram,
    }


def _base_checkpoint(path: Path, salt: str = "") -> Path:
    path.mkdir()
    weight_map = {
        f"model.language_model.layers.{layer}.self_attn.k_proj.weight": "model-bf16-1"
        for layer in QSA_LAYERS
    }
    weight_map["mtp.layers.0.self_attn.k_proj.weight" + salt] = "model-bf16-1"
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}) + "\n"
    )
    return path


def _report(tmp_path: Path) -> Path:
    """summarize over traces in the observer's format, two ranks."""
    traces = tmp_path / "calib"
    traces.mkdir()
    for rank in range(2):
        with (traces / f"qsa-kv-rank{rank}-pid{rank}.jsonl").open("w") as stream:
            for layer in [*QSA_LAYERS, MTP_LAYER]:
                record = {
                    "schema_version": 1,
                    "layer_id": layer,
                    "corpus_shard": "replay",
                    "histogram": {"bins": 4096, "log2_min": -24.0, "log2_max": 16.0},
                    # The larger maximum is on rank 1.
                    "k": _stat(_max_abs(layer, "k") - (1 - rank)),
                    "v": _stat(_max_abs(layer, "v") - (1 - rank)),
                }
                stream.write(json.dumps(record) + "\n")
    output = tmp_path / "kv-report-13.json"
    calibration.summarize(
        argparse.Namespace(
            input_dir=[str(traces)],
            exclude_corpus_shard=[],
            output=str(output),
            expected_layers=13,
        )
    )
    return output


def _target_pack(tmp_path: Path, base: Path, report: Path) -> tuple[Path, Path]:
    target = tmp_path / "kv-report-target.json"
    scale_pack.target_report(
        argparse.Namespace(
            report=str(report), output=str(target), mtp_layer_id=MTP_LAYER
        )
    )
    pack = tmp_path / "pack"
    if importlib.util.find_spec("safetensors") is not None:
        calibration.build_overlay(
            argparse.Namespace(
                base_checkpoint=str(base),
                report=str(target),
                output_dir=str(pack),
                expected_layers=12,
                unit_scale_negative_control=False,
            )
        )
    else:
        pack.mkdir()
        tensors = json.loads(target.read_text())["tensors"]
        materializer.save_scale_shard(
            pack / scale_pack.SCALE_FILENAME,
            {
                name.replace("model.", "model.language_model.", 1): details["scale"]
                for name, details in tensors.items()
            },
            {"format": "pt"},
        )
    return pack, target


def _manifest(base: Path, pack: Path, target: Path) -> None:
    scale_pack.manifest(
        argparse.Namespace(
            pack_dir=str(pack),
            base_checkpoint=str(base),
            report=str(target),
            artifact_id="sx-test",
        )
    )


def _materialize(base: Path, pack: Path, output: Path, report: Path) -> None:
    materializer.materialize(
        argparse.Namespace(
            base_checkpoint=str(base),
            output_dir=str(output),
            pack_dir=str(pack),
            mtp_report=str(report),
            mtp_layer_id=MTP_LAYER,
            mtp_scale_from_layer=None,
            mtp_unit_scale=False,
            target_scale_source="published",
            target_report=None,
        )
    )


def _read_scales(path: Path) -> dict[str, float]:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
        header.pop("__metadata__", None)
        values = {}
        for name, details in header.items():
            assert details["dtype"] == "F32" and details["shape"] in ([], [1])
            handle.seek(8 + size + details["data_offsets"][0])
            values[name] = struct.unpack("<f", handle.read(4))[0]
    return values


def test_own_calibration_becomes_the_served_overlay(tmp_path):
    base = _base_checkpoint(tmp_path / "base")
    report = _report(tmp_path)
    assert json.loads(report.read_text())["tensor_count"] == 26
    pack, target = _target_pack(tmp_path, base, report)
    assert json.loads(target.read_text())["qsa_layer_ids"] == QSA_LAYERS
    _manifest(base, pack, target)
    final = tmp_path / "final"
    _materialize(base, pack, final, report)

    index = json.loads((final / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    target_scales = _read_scales(final / "model-kvscales.safetensors")
    mtp_scales = _read_scales(final / "model-bf16-kvscales-mtp.safetensors")
    assert len(target_scales) == 24 and len(mtp_scales) == 2
    for layer in QSA_LAYERS:
        for kind in ("k", "v"):
            name = f"model.language_model.layers.{layer}.self_attn.{kind}_scale"
            assert weight_map[name] == "model-kvscales.safetensors"
            expected = calibration._ceil_float32(_max_abs(layer, kind) / E4M3_MAX)
            assert target_scales[name] == expected
            assert target_scales[name] * E4M3_MAX >= _max_abs(layer, kind)
    for kind in ("k", "v"):
        name = f"mtp.layers.0.self_attn.{kind}_scale"
        assert weight_map[name] == "model-bf16-kvscales-mtp.safetensors"
        expected = calibration._ceil_float32(_max_abs(MTP_LAYER, kind) / E4M3_MAX)
        assert mtp_scales[name] == expected
    provenance = json.loads((final / "kvscales-provenance.json").read_text())
    assert provenance["artifact_id"] == "sx-test"
    assert provenance["mtp_scale_source"] == "calibrated"
    assert provenance["target_scale_source"] == "published"


def test_pack_is_bound_to_its_base_checkpoint(tmp_path):
    base = _base_checkpoint(tmp_path / "base")
    report = _report(tmp_path)
    pack, target = _target_pack(tmp_path, base, report)
    _manifest(base, pack, target)
    other = _base_checkpoint(tmp_path / "other", salt=".other")
    with pytest.raises(ValueError, match="revision does not match the scale pack"):
        _materialize(other, pack, tmp_path / "refused", report)
    assert not (tmp_path / "refused").exists()


def test_target_report_needs_the_mtp_layer(tmp_path):
    report = _report(tmp_path)
    args = argparse.Namespace(
        report=str(report), output=str(tmp_path / "target.json"), mtp_layer_id=49
    )
    with pytest.raises(ValueError, match="MTP layer 49 in the report, found 0"):
        scale_pack.target_report(args)
    assert not (tmp_path / "target.json").exists()
    args.mtp_layer_id = MTP_LAYER
    scale_pack.target_report(args)
    with pytest.raises(FileExistsError):
        scale_pack.target_report(args)


def test_manifest_rejects_a_report_for_other_tensors(tmp_path):
    base = _base_checkpoint(tmp_path / "base")
    report = _report(tmp_path)
    pack, target = _target_pack(tmp_path, base, report)
    with pytest.raises(ValueError, match="cover different tensors"):
        _manifest(base, pack, report)  # the 13-layer report
    assert not (pack / "kvscales-manifest.json").exists()
    _manifest(base, pack, target)
    with pytest.raises(FileExistsError):
        _manifest(base, pack, target)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
