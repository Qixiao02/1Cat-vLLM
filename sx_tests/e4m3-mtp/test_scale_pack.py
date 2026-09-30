# SPDX-License-Identifier: Apache-2.0
"""Scale overlay for a checkpoint of our own, from traces to the served view.

Run (CPU, no vLLM install and no GPU needed):

    python -m pytest -q sx_tests/e4m3-mtp/test_scale_pack.py

Runs the offline part of the calibration procedure on synthetic observer
traces of 12 target QSA layers plus the MTP layer (id 48):

    qsa_kv_calibration.py summarize --expected-layers 13
    sx_scale_pack.py target-report
    qsa_kv_calibration.py overlay        (needs torch + safetensors; without
                                          safetensors its scale shard is
                                          written by the materializer's own
                                          writer, same layout)
    sx_scale_pack.py manifest
    materialize_qsa_scale_overlay.py --mtp-report

The base checkpoint has the layout of Swift-1.5-Qwen3.8-Flash-Next-NVFP4:
48 model-layer-NNNNN shards, 10 model-plefp8-NNNNN shards that are symlinks
into another conversion, one model-tail shard holding the 31 mtp.* tensors,
and no model-bf16-* shard. The overlay is then read the way the engine reads
it: the file patterns of the drafter (Qwen4ExpMTP.allow_patterns_overrides)
and of the target (the default *.safetensors), the first pattern with a match
wins, and the index filters the files.

Asserted: every scale is max_abs / 448 rounded up to FP32 (target) or as
reported (MTP); the MTP scale shard gets a name outside model-bf16-* on this
layout, so the drafter still sees all 31 mtp.* tensors plus its 2 scales and
the target all base tensors plus its 24; a shard name that would hide either
is refused; every shard of the merged index exists, holds exactly the tensors
the index assigns to it, and the PLE links still end in the other conversion;
the pack is refused for a checkpoint with a different index and for a report
that does not match the scale shard.
Expected: all pass. The layout tests create symlinks and are skipped where
the OS refuses that (Windows without the privilege); the other tests use a
base that holds nothing but its index.
"""

from __future__ import annotations

import argparse
import fnmatch
import glob
import importlib.util
import json
import os
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e4m3_boot  # noqa: E402

e4m3_boot.provide_regex()
REPO = Path(e4m3_boot.REPO)
TOOLS = REPO / "tools" / "qwen4_exp"
QSA_LAYERS = list(range(3, 48, 4))
MTP_LAYER = 48
E4M3_MAX = 448.0
TARGET_SHARD = "model-kvscales.safetensors"
MTP_SHARD = "model-kvscales-mtp.safetensors"
# DefaultModelLoader._prepare_weights, load_format "hf" with the .pt fallback.
TARGET_PATTERNS = ["*.safetensors", "*.bin", "*.pt"]


def _tool(name: str):
    spec = importlib.util.spec_from_file_location(f"sx_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


calibration = _tool("qsa_kv_calibration")
materializer = _tool("materialize_qsa_scale_overlay")
scale_pack = _tool("sx_scale_pack")


def _loader():
    """The drafter's patterns and name remap and the loader's index filter.

    Cut from the sources next to this checkout (none of the three needs
    torch); where only the tests are mounted, the installed vLLM provides them.
    """
    mtp = REPO / "vllm" / "models" / "qwen4_exp" / "nvidia" / "mtp.py"
    utils = REPO / "vllm" / "model_executor" / "model_loader" / "weight_utils.py"
    if not mtp.is_file():
        from vllm.model_executor.model_loader import weight_utils
        from vllm.models.qwen4_exp.nvidia import mtp as installed

        return (
            installed.Qwen4ExpMTP.allow_patterns_overrides,
            installed._remap_mtp_weight_name,
            weight_utils.filter_duplicate_safetensors_files,
        )
    namespace = {"nn": SimpleNamespace(Module=object), "os": os, "json": json}
    source = e4m3_boot._cut(
        str(mtp),
        ("_remap_mtp_weight_name",),
        {"Qwen4ExpMTP": ("allow_patterns_overrides",)},
    ) + e4m3_boot._cut(str(utils), ("filter_duplicate_safetensors_files",))
    exec(compile(source, str(mtp), "exec"), namespace)  # noqa: S102
    return (
        namespace["Qwen4ExpMTP"].allow_patterns_overrides,
        namespace["_remap_mtp_weight_name"],
        namespace["filter_duplicate_safetensors_files"],
    )


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


def _write_index(path: Path, weight_map: dict[str, str]) -> None:
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}) + "\n"
    )


def _base_checkpoint(path: Path, salt: str = "") -> Path:
    """A base that is nothing but its index (no entry to link)."""
    path.mkdir()
    weight_map = {
        f"model.language_model.layers.{layer}.self_attn.k_proj.weight": "model-layer"
        for layer in QSA_LAYERS
    }
    weight_map["mtp.layers.0.self_attn.k_proj.weight" + salt] = "model-tail"
    _write_index(path, weight_map)
    return path


MTP_TENSORS = [
    "mtp.fc_embedding.weight",
    "mtp.fc_hidden.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight",
    "mtp.hyper_connection_mixer.norm.weight",
    *(
        f"mtp.layers.0.{name}"
        for name in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "self_attn.indexer.q_proj.weight",
            "self_attn.indexer.k_proj.weight",
            "self_attn.indexer.k_norm.weight",
            "attn_hyper_connection.input_mix_weight_down.weight",
            "attn_hyper_connection.input_mix_weight_up.weight",
            "attn_hyper_connection.block_inject_weight.weight",
            "attn_hyper_connection.norm.weight",
            "mlp_hyper_connection.input_mix_weight_down.weight",
            "mlp_hyper_connection.input_mix_weight_up.weight",
            "mlp_hyper_connection.block_inject_weight.weight",
            "mlp_hyper_connection.norm.weight",
            "mlp.gate.weight",
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
            "mlp.shared_expert.gate_proj.weight",
            "mlp.shared_expert.up_proj.weight",
            "mlp.shared_expert.down_proj.weight",
            "mlp.shared_expert_gate.weight",
        )
    ),
]
assert len(MTP_TENSORS) == 31


def _swift_layout(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """48 layer shards, 10 symlinked PLE shards, one tail shard, no bf16 shard."""
    models = tmp_path / "models"
    base, other = models / "Swift-NVFP4", models / "Other-NVFP4"
    base.mkdir(parents=True)
    other.mkdir()
    try:
        (base / "probe").symlink_to("../Other-NVFP4")
    except OSError:
        pytest.skip("symlinks cannot be created here")
    (base / "probe").unlink()
    shards: dict[str, list[str]] = {}
    for layer in range(48):
        prefix = f"model.language_model.layers.{layer}"
        names = [f"{prefix}.mlp.experts.0.down_proj.weight"]
        if layer in QSA_LAYERS:
            names += [f"{prefix}.self_attn.{p}_proj.weight" for p in "qkvo"]
        else:
            names += [f"{prefix}.linear_attn.in_proj_qkv.weight"]
        shards[f"model-layer-{layer:05d}.safetensors"] = names
    for shard in range(10):
        shards[f"model-plefp8-{shard:05d}.safetensors"] = [
            f"model.language_model.ple.tables.{shard}.weight"
        ]
    shards["model-tail.safetensors"] = [
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
        *MTP_TENSORS,
    ]
    weight_map = {}
    for filename, names in shards.items():
        ple = filename.startswith("model-plefp8-")
        materializer.save_scale_shard(
            (other if ple else base) / filename, dict.fromkeys(names, 1.0), {}
        )
        if ple:
            (base / filename).symlink_to(f"../Other-NVFP4/{filename}")
        weight_map.update(dict.fromkeys(names, filename))
    (base / "config.json").write_text("{}\n")
    _write_index(base, weight_map)
    return base, weight_map


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


def _materialize(base: Path, pack: Path, output: Path, report: Path, **extra) -> None:
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
            **extra,
        )
    )


def _overlay(tmp_path: Path, base: Path, **extra) -> Path:
    report = _report(tmp_path)
    pack, target = _target_pack(tmp_path, base, report)
    _manifest(base, pack, target)
    final = base.parent / f"{base.name}-e4m3kv"
    _materialize(base, pack, final, report, **extra)
    return final


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


def _loaded_tensors(folder: Path, patterns: list[str], index_filter) -> dict[str, str]:
    """Tensor name -> shard, as DefaultModelLoader._prepare_weights picks files:
    the first pattern that matches anything, then only files the index names."""
    files: list[str] = []
    for pattern in patterns:
        files += glob.glob(os.path.join(str(folder), pattern))
        if files:
            break
    files = index_filter(files, str(folder), "model.safetensors.index.json")
    return {
        name: os.path.basename(path)
        for path in files
        for name in scale_pack._shard_tensor_names(Path(path))
    }


def _check_scales(final: Path, mtp_shard: str) -> None:
    weight_map = json.loads((final / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    target_scales = _read_scales(final / TARGET_SHARD)
    mtp_scales = _read_scales(final / mtp_shard)
    assert len(target_scales) == 24 and len(mtp_scales) == 2
    for layer in QSA_LAYERS:
        for kind in ("k", "v"):
            name = f"model.language_model.layers.{layer}.self_attn.{kind}_scale"
            assert weight_map[name] == TARGET_SHARD
            expected = calibration._ceil_float32(_max_abs(layer, kind) / E4M3_MAX)
            assert target_scales[name] == expected
            assert target_scales[name] * E4M3_MAX >= _max_abs(layer, kind)
    for kind in ("k", "v"):
        name = f"mtp.layers.0.self_attn.{kind}_scale"
        assert weight_map[name] == mtp_shard
        expected = calibration._ceil_float32(_max_abs(MTP_LAYER, kind) / E4M3_MAX)
        assert mtp_scales[name] == expected
    provenance = json.loads((final / "kvscales-provenance.json").read_text())
    assert provenance["artifact_id"] == "sx-test"
    assert provenance["mtp_scale_file"] == mtp_shard
    assert provenance["mtp_scale_source"] == "calibrated"
    assert provenance["target_scale_source"] == "published"


def test_own_calibration_becomes_the_served_overlay(tmp_path):
    base = _base_checkpoint(tmp_path / "base")
    final = _overlay(tmp_path, base)
    _check_scales(final, MTP_SHARD)


def test_swift_layout_overlay_loads_for_drafter_and_target(tmp_path):
    base, base_map = _swift_layout(tmp_path)
    final = _overlay(tmp_path, base)
    _check_scales(final, MTP_SHARD)
    draft_patterns, remap, index_filter = _loader()
    assert draft_patterns[0] == materializer.DRAFT_SHARD_PATTERN

    # The directory: every base entry linked, two scale shards, index, provenance.
    assert sorted(entry.name for entry in final.iterdir()) == sorted(
        [
            *(entry.name for entry in base.iterdir()),
            TARGET_SHARD,
            MTP_SHARD,
            "kvscales-provenance.json",
        ]
    )
    assert not list(final.glob(draft_patterns[0]))
    for entry in base.iterdir():
        if entry.name != "model.safetensors.index.json":
            assert (final / entry.name).is_symlink()
            assert (final / entry.name).resolve() == entry.resolve()
    ple = (final / "model-plefp8-00003.safetensors").resolve()
    assert ple.parent.name == "Other-NVFP4" and ple.is_file()

    # The index: the base map plus 26 scales, and each shard holds exactly
    # the tensors the index assigns to it.
    index = json.loads((final / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    assert len(weight_map) == len(base_map) + 26
    assert {k: weight_map[k] for k in base_map} == base_map
    by_shard: dict[str, set[str]] = {}
    for name, filename in weight_map.items():
        by_shard.setdefault(filename, set()).add(name)
    for filename, names in by_shard.items():
        assert set(scale_pack._shard_tensor_names(final / filename)) == names

    # The drafter falls through to *.safetensors, as on the base, and maps
    # its weights and its two scales onto its modules.
    assert [p for p in draft_patterns if glob.glob(str(final / p))][0] == (
        "*.safetensors"
    )
    seen = _loaded_tensors(final, draft_patterns, index_filter)
    assert set(seen) == set(weight_map)
    drafter = {remap(name) for name in seen} - {None}
    assert {remap(name) for name in MTP_TENSORS} <= drafter
    assert {
        "model.layers.0.self_attn.k_scale",
        "model.layers.0.self_attn.v_scale",
    } <= drafter
    assert not any("language_model" in name for name in drafter)
    before = _loaded_tensors(base, draft_patterns, index_filter)
    assert {remap(name) for name in before} - {None} == drafter - {
        "model.layers.0.self_attn.k_scale",
        "model.layers.0.self_attn.v_scale",
    }

    # The target reads every shard; it skips "mtp." names itself.
    seen = _loaded_tensors(final, TARGET_PATTERNS, index_filter)
    assert set(seen) == set(weight_map)
    scales = {name for name in seen if name.endswith("_scale") and "mtp." not in name}
    assert len(scales) == 24 and {seen[name] for name in scales} == {TARGET_SHARD}


def test_shard_name_that_hides_the_mtp_weights_is_refused(tmp_path):
    base, _ = _swift_layout(tmp_path)
    draft_patterns, remap, index_filter = _loader()
    with pytest.raises(ValueError, match="would not load with this base"):
        _overlay(tmp_path, base, mtp_scale_filename=materializer.MTP_SCALE_FILENAME)
    assert not (base.parent / f"{base.name}-e4m3kv").exists()

    # What the refused name would have done: the first pattern matches the
    # scale shard alone.
    (base / materializer.MTP_SCALE_FILENAME).write_bytes(b"")
    first = [p for p in draft_patterns if glob.glob(str(base / p))][0]
    assert first == materializer.DRAFT_SHARD_PATTERN
    assert [os.path.basename(p) for p in glob.glob(str(base / first))] == [
        materializer.MTP_SCALE_FILENAME
    ]


def test_shard_name_follows_the_base_layout(tmp_path):
    def name(entries, **extra):
        base = tmp_path / f"base{len(list(tmp_path.iterdir()))}"
        base.mkdir()
        for entry in entries:
            (base / entry).write_bytes(b"")
        return materializer._mtp_scale_filename(
            argparse.Namespace(**extra), base, TARGET_SHARD
        )

    bf16 = ["model-bf16-00001-of-00004.safetensors", "model-00001.safetensors"]
    swift = ["model-layer-00000.safetensors", "model-tail.safetensors"]
    assert name(bf16) == materializer.MTP_SCALE_FILENAME
    assert fnmatch.fnmatch(name(bf16), materializer.DRAFT_SHARD_PATTERN)
    assert name(swift) == MTP_SHARD
    assert not fnmatch.fnmatch(MTP_SHARD, materializer.DRAFT_SHARD_PATTERN)
    assert name(swift, mtp_scale_filename="kv-mtp.safetensors") == "kv-mtp.safetensors"
    override = "model-bf16-mtp-scales.safetensors"
    assert name(bf16, mtp_scale_filename=override) == override
    for entries, bad in (
        (bf16, MTP_SHARD),  # the drafter would never read it
        (swift, override),  # the drafter would read nothing else
        (swift, TARGET_SHARD),
        (swift, "sub/kv-mtp.safetensors"),
        (swift, "kv-mtp.bin"),
    ):
        with pytest.raises(ValueError):
            name(entries, mtp_scale_filename=bad)


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
