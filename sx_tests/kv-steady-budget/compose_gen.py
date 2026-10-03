# SPDX-License-Identifier: Apache-2.0
"""Write the docker compose file for one KV-steady-budget trial on the V100 box.

Standard library only, so ``run_on_v100.sh`` can call it on the server and the
CPU tests can import it. Nothing is read from a production compose file: every
line comes from the arguments, so the trial is reproducible on any machine that
has the image, the model directory and four free GPUs.

The serve flags are the native-MTP arm of the 2026-10-01 comparison
(sx_bench/as_run/arm_compose.py, arm FM) with the utilisation, the lane and the
switch as parameters::

    --tensor-parallel-size 4 --dtype half --attention-backend FLASH_ATTN_V100
    --max-model-len 32768 --max-num-seqs 16 --max-num-batched-tokens 8192
    --gpu-memory-utilization <u> --kv-cache-dtype auto --trust-remote-code
    --enable-prefix-caching --enable-chunked-prefill --enable-auto-tool-choice
    --tool-call-parser qwen3_coder --reasoning-parser qwen3 --language-model-only
    --speculative-config {"method":"mtp","num_speculative_tokens":4}

The no-MTP lane drops the speculative config and takes the production shape of
that lane (max-num-seqs 24, max-model-len 131072, MoE tune tokens 240).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

SERVED_NAME = "Swift-1.5-Qwen3.8-Flash-Next"
MODEL_SUBDIR = "Swift-1.5-Qwen3.8-Flash-Next-NVFP4"
MTP_CONFIG = {"method": "mtp", "num_speculative_tokens": 4}

# Environment shared by both lanes (arm_compose.py COMMON_ENV + FORK_ENV).
COMMON_ENV = {
    "NVIDIA_VISIBLE_DEVICES": "all",
    "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
    "VLLM_QWEN4EXP_PLE_HOST_GIB": "12",
    "VLLM_ENGINE_READY_TIMEOUT_S": "1800",
    "HOME": "/cache/home",
    "XDG_CACHE_HOME": "/cache",
    "HF_HOME": "/cache/huggingface",
    "HUGGINGFACE_HUB_CACHE": "/cache/huggingface/hub",
    "TORCH_HOME": "/cache/torch",
    "TRITON_CACHE_DIR": "/cache/triton",
    "TILELANG_CACHE_DIR": "/cache/tilelang",
    "USER": "sx-inference",
    "LOGNAME": "sx-inference",
    "OMP_NUM_THREADS": "8",
    "VLLM_SM70_QWEN38_HYBRID_PLE": "0",
    "VLLM_PLE_CPU_OFFLOAD": "0",
    "VLLM_PLE_DISK_OFFLOAD": "0",
    "VLLM_SM70_NVFP4_MOE_GROUPED_DECODE": "1",
}

# Where the image's vLLM lives (the layout of the fork's V100 images; see
# sx_bench/as_run/mtp_compose.py, which overlays files the same way).
DEFAULT_SITE_PACKAGES = "/opt/venv/lib/python3.12/site-packages"

LANES = {
    "MTP": {
        "max_num_seqs": 16,
        "max_model_len": 32768,
        "moe_tune_tokens": "1200",
        "speculative": True,
        "utils": "0.87 0.90 0.93 0.95",
    },
    "no-MTP": {
        "max_num_seqs": 24,
        "max_model_len": 131072,
        "moe_tune_tokens": "240",
        "speculative": False,
        "utils": "0.90 0.93 0.95",
    },
}


def normalise_lane(value: str) -> str:
    """Accept MTP / mtp / no-MTP / nomtp / no_mtp."""
    key = value.strip().lower().replace("_", "-")
    if key == "mtp":
        return "MTP"
    if key in ("no-mtp", "nomtp"):
        return "no-MTP"
    raise ValueError(f"lane must be MTP or no-MTP, got {value!r}")


def build_env(
    lane: str,
    switch: str,
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    preset = LANES[lane]
    env = dict(COMMON_ENV)
    env["VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS"] = preset["moe_tune_tokens"]
    if switch != "auto":  # "auto": leave it unset and let the engine decide
        env["SX_OPT_KV_STEADY_BUDGET"] = switch
    if extra:
        env.update(extra)
    return env


def build_flags(
    lane: str,
    util: str,
    *,
    max_num_seqs: int | None = None,
    max_model_len: int | None = None,
    max_num_batched_tokens: int = 8192,
    extra_flags: list[str] | None = None,
) -> list[str]:
    preset = LANES[lane]
    flags = [
        f"/models/{MODEL_SUBDIR}",
        f"--served-model-name={SERVED_NAME}",
        "--host=0.0.0.0",
        "--port=8001",
        "--tensor-parallel-size=4",
        "--dtype=half",
        "--attention-backend=FLASH_ATTN_V100",
        f"--max-model-len={max_model_len or preset['max_model_len']}",
        f"--max-num-seqs={max_num_seqs or preset['max_num_seqs']}",
        f"--max-num-batched-tokens={max_num_batched_tokens}",
        f"--gpu-memory-utilization={util}",
        "--kv-cache-dtype=auto",
        "--trust-remote-code",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--enable-auto-tool-choice",
        "--tool-call-parser=qwen3_coder",
        "--reasoning-parser=qwen3",
        "--language-model-only",
    ]
    if preset["speculative"]:
        flags.append("--speculative-config=" + json.dumps(MTP_CONFIG, separators=(",", ":")))
    flags.extend(extra_flags or [])
    return flags


def _q(text: str) -> str:
    """Double-quoted YAML scalar (JSON strings are valid YAML)."""
    return json.dumps(text)


def overlay_files(patch_dir: str) -> dict[str, str]:
    """``{path relative to patch_dir: absolute host path}`` of every file under
    ``patch_dir/vllm`` (sorted), for a read-only overlay of the image's package.
    A change that only touches Python needs no image build: copy the changed
    files into ``patch_dir/vllm/...`` and the engine runs them."""
    root = os.path.join(patch_dir, "vllm")
    if not os.path.isdir(root):
        raise ValueError(f"{patch_dir} has no vllm/ directory to overlay")
    files: dict[str, str] = {}
    for base, _, names in os.walk(root):
        for name in names:
            if name.endswith((".pyc", ".pyo")) or "__pycache__" in base:
                continue
            path = os.path.join(base, name)
            files[os.path.relpath(path, patch_dir).replace(os.sep, "/")] = os.path.abspath(
                path
            )
    if not files:
        raise ValueError(f"no files under {root}")
    return dict(sorted(files.items()))


def render(
    *,
    image: str,
    tag: str,
    lane: str,
    util: str,
    switch: str,
    gpus: str,
    models_dir: str,
    cache_dir: str,
    port: int,
    max_num_seqs: int | None = None,
    max_model_len: int | None = None,
    max_num_batched_tokens: int = 8192,
    extra_env: dict[str, str] | None = None,
    extra_flags: list[str] | None = None,
    shm: str = "16gb",
    overlay: dict[str, str] | None = None,
    site_packages: str = DEFAULT_SITE_PACKAGES,
) -> str:
    if switch not in ("0", "1", "auto"):
        raise ValueError(f"switch must be 0, 1 or auto, got {switch!r}")
    device_ids = [g.strip() for g in gpus.split(",") if g.strip()]
    if len(device_ids) != 4:
        raise ValueError(f"the lane runs TP4: --gpus needs four ids, got {gpus!r}")
    env = build_env(lane, switch, extra_env)
    flags = build_flags(
        lane,
        util,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        extra_flags=extra_flags,
    )
    lines = [
        f"# KV steady budget trial {tag}: lane {lane}, util {util}, "
        "SX_OPT_KV_STEADY_BUDGET="
        + ("(unset: the engine's default)" if switch == "auto" else switch)
        + ". Generated by compose_gen.py; temporary.",
        f"name: {_q('sx-kvsteady-' + tag)}",
        "services:",
        "  engine:",
        f"    image: {_q(image)}",
        f"    container_name: {_q('sx-kvsteady-' + tag)}",
        '    restart: "no"',
        "    ipc: host",
        f"    shm_size: {_q(shm)}",
        "    cap_add: [IPC_LOCK]",
        "    ulimits:",
        "      memlock: -1",
        "      stack: 67108864",
        f"    ports: [{_q(f'127.0.0.1:{port}:8001')}]",
        "    volumes:",
        f"      - {{type: bind, source: {_q(models_dir)}, target: /models, read_only: true}}",
        f"      - {{type: bind, source: {_q(cache_dir)}, target: /cache}}",
    ]
    for rel, host in (overlay or {}).items():
        lines.append(
            f"      - {{type: bind, source: {_q(host)}, "
            f"target: {_q(site_packages + '/' + rel)}, read_only: true}}"
        )
    lines.append("    environment:")
    for key, value in env.items():
        lines.append(f"      {key}: {_q(value)}")
    lines += [
        "    deploy:",
        "      resources:",
        "        reservations:",
        "          devices:",
        "            - driver: nvidia",
        "              device_ids: [" + ", ".join(_q(d) for d in device_ids) + "]",
        "              capabilities: [gpu]",
        "    healthcheck:",
        '      test: ["CMD-SHELL", "curl -sf http://127.0.0.1:8001/health >/dev/null || exit 1"]',
        "      interval: 30s",
        "      timeout: 10s",
        "      retries: 3",
        "      start_period: 3600s",
        '    entrypoint: ["vllm", "serve"]',
        "    command:",
    ]
    for flag in flags:
        lines.append(f"      - {_q(flag)}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "--lane-info":
        # run_on_v100.sh: "<normalised lane>|<default utilisations>"
        try:
            lane = normalise_lane(argv[1])
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        print(f"{lane}|{LANES[lane]['utils']}")
        return 0
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--image", required=True)
    parser.add_argument("--tag", required=True, help="suffix of the container name")
    parser.add_argument("--lane", default="MTP", help="MTP or no-MTP")
    parser.add_argument("--util", required=True)
    parser.add_argument("--switch", default="1", choices=("0", "1", "auto"))
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--models-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--port", type=int, default=8141)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--env", action="append", default=[], metavar="K=V")
    parser.add_argument("--serve-arg", action="append", default=[], metavar="FLAG")
    parser.add_argument(
        "--patch-dir",
        help="overlay every file of PATCH_DIR/vllm/ read-only over the image's package",
    )
    parser.add_argument("--site-packages", default=DEFAULT_SITE_PACKAGES)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    extra_env: dict[str, str] = {}
    for item in args.env:
        key, sep, value = item.partition("=")
        if not sep or not key:
            parser.error(f"--env needs K=V, got {item!r}")
        extra_env[key] = value
    text = render(
        image=args.image,
        tag=args.tag,
        lane=normalise_lane(args.lane),
        util=args.util,
        switch=args.switch,
        gpus=args.gpus,
        models_dir=args.models_dir,
        cache_dir=args.cache_dir,
        port=args.port,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        extra_env=extra_env,
        extra_flags=args.serve_arg,
        overlay=overlay_files(args.patch_dir) if args.patch_dir else None,
        site_packages=args.site_packages,
    )
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
