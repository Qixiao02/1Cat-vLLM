#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cold/warm torch.compile-cache parity harness for the V100 server.

Starts the engine several times against controlled cache directories and
compares what the cache is allowed to change: nothing but the startup time.

    arm        SX_OPT_COMPILE_CACHE  cache dir      what it tells you
    off        0 (today)             cache-off      reference; time to healthy
    cold       <mode>                cache-on, new  the compile with saving; is it
                                                    the same model as off?
    warm       <mode>                cache-on, same the reload; reuse really happened
                                                    and the outputs did not move
    off2       0                     cache-off      control: is the reference itself
    (optional)                                      reproducible across restarts?
    off_noaot  0 + VLLM_USE_AOT_COMPILE=0           control: AOT vs non-AOT numerics
    (optional)                       cache-off      without any reload
    warm2      <mode>                cache-on, same a second reload (reload of a
    (optional)                                      reloaded artifact)
    inval      <mode> + an unused    cache-on, same must NOT reuse: proves a changed
    (optional) VLLM_ switch                         switch changes the cache key

For every arm it records the time to healthy, the compile/capture phase times
and the reuse counters from the engine log, greedy outputs (token ids) of a
fixed prompt set (thinking off, temperature 0, 256-512 tokens, 8K-32K prompts
included) and, with MTP, the draft acceptance counters. It exits 1, loudly, if
any compared pair of arms differs by a single token or a counter, if the warm
arm did not reuse what the cold arm saved, or if the warm log shows a load
failure, a stale pointer or a failed save.

    python3 cache_parity.py run --compose BASE.yaml --image IMG --lane nomtp
    python3 cache_parity.py compare --arm off=a.json --arm cold=b.json ...
    python3 cache_parity.py prompts --out prompts.json --tokenize-port 8141

Only the standard library is used. The default engine launcher turns a compose
file of the shape of ``compose.swift15-flashnext-tp4-gpu0123.yaml`` into a
``vllm serve`` arm exactly as ``sx_bench/as_run/arm_compose.py`` does; any other
launcher can be plugged in with ``--engine-script`` (see README.md).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any

MODEL_PATH = "/models/Swift-1.5-Qwen3.8-Flash-Next-NVFP4"
SERVED_NAME = "Swift-1.5-Qwen3.8-Flash-Next"
MTP_CONFIG = '{"method":"mtp","num_speculative_tokens":4}'

# ---------------------------------------------------------------------------
# lane presets (the production flags and the switches of the 8031 compose;
# the MTP lane is the "FM" arm of sx_bench/as_run/arm_compose.py)
# ---------------------------------------------------------------------------

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
}
FORK_ENV = {
    "OMP_NUM_THREADS": "8",
    "VLLM_SM70_QWEN38_HYBRID_PLE": "0",
    "VLLM_PLE_CPU_OFFLOAD": "0",
    "VLLM_PLE_DISK_OFFLOAD": "0",
    "VLLM_SM70_NVFP4_MOE_GROUPED_DECODE": "1",
    "VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS": "240",
}
LANES: dict[str, dict[str, Any]] = {
    "nomtp": dict(
        env=dict(FORK_ENV),
        seqs=24,
        util="0.90",
        maxlen=131072,
        mtp=False,
        long_lengths=(8000, 16000, 32000),
    ),
    "mtp": dict(
        env={**FORK_ENV, "VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS": "1200"},
        seqs=16,
        util="0.87",
        maxlen=32768,
        mtp=True,
        long_lengths=(8000, 16000, 28000),
    ),
}

# Log lines that fail the run when they appear in a warm (cache-on) engine log.
BAD_LOG_PATTERNS = (
    "Compiling model again due to a load failure",
    "unable to save AOT compiled function",
    "illegal memory access",
    "resides on host memory",
    "not registered with any CUDA device",
    "Kernel index",
    "no Triton side table",
    "Source code has changed since the last compilation",
    "The compiled artifact is not serializable",
)

# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ArmSpec:
    name: str
    switch: str  # value of SX_OPT_COMPILE_CACHE ("0" = today)
    cache: str  # "off" or "on": which cache directory
    fresh: bool = False  # wipe (and re-seed) that cache directory first
    extra_env: tuple[tuple[str, str], ...] = ()


def arm_table(mode: str) -> dict[str, ArmSpec]:
    force_load = (("VLLM_FORCE_AOT_LOAD", "1"),) if mode == "aot" else ()
    return {
        "off": ArmSpec("off", "0", "off", fresh=True),
        "off2": ArmSpec("off2", "0", "off"),
        "off_noaot": ArmSpec(
            "off_noaot", "0", "off", extra_env=(("VLLM_USE_AOT_COMPILE", "0"),)
        ),
        "cold": ArmSpec("cold", mode, "on", fresh=True),
        "warm": ArmSpec("warm", mode, "on", extra_env=force_load),
        "warm2": ArmSpec("warm2", mode, "on", extra_env=force_load),
        # an unregistered VLLM_ switch that changes nothing in the engine: the
        # cache key must still change, so nothing may be reused
        "inval": ArmSpec("inval", mode, "on", extra_env=(("VLLM_SX_CACHE_PROBE", "1"),)),
    }


# pairs whose outputs must be identical (reference first) and control pairs
GATE_PAIRS = (
    ("off", "cold"),
    ("off", "warm"),
    ("cold", "warm"),
    ("warm", "warm2"),
    ("cold", "inval"),
)
CONTROL_PAIRS = (("off", "off2"), ("off", "off_noaot"))
DEFAULT_ARMS = ("off", "cold", "warm")


# ---------------------------------------------------------------------------
# compose generation (arm_compose.py, generalised)
# ---------------------------------------------------------------------------


def _sub(text: str, pattern: str, repl: str, count: int = 1, required: bool = True):
    new, n = re.subn(pattern, lambda m: repl, text, count=count, flags=re.M)
    if required and n != count:
        raise ValueError(
            "compose template does not match %r (%d matches, expected %d); the "
            "base compose must look like compose.swift15-flashnext-tp4-*.yaml or "
            "use --engine-script" % (pattern, n, count)
        )
    return new


def serve_flags(
    lane: dict[str, Any],
    port: int,
    drop: Sequence[str] = (),
    extra: Sequence[str] = (),
) -> list[str]:
    flags = [
        MODEL_PATH,
        "--served-model-name=%s" % SERVED_NAME,
        "--host=0.0.0.0",
        "--port=8001",
        "--tensor-parallel-size=4",
        "--dtype=half",
        "--attention-backend=FLASH_ATTN_V100",
        "--max-model-len=%d" % lane["maxlen"],
        "--max-num-seqs=%d" % lane["seqs"],
        "--max-num-batched-tokens=8192",
        "--gpu-memory-utilization=%s" % lane["util"],
        "--kv-cache-dtype=auto",
        "--trust-remote-code",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--enable-auto-tool-choice",
        "--tool-call-parser=qwen3_coder",
        "--reasoning-parser=qwen3",
        "--language-model-only",
    ]
    if lane["mtp"]:
        flags.append("--speculative-config=" + MTP_CONFIG)
    flags = [f for f in flags if not any(f.startswith(d) for d in drop)]
    flags.extend(extra)
    return flags


def compose_from_base(
    base: str,
    *,
    name: str,
    container: str,
    image: str | None,
    port: int,
    cache_dir: str,
    gpus: Sequence[str],
    env: dict[str, str],
    flags: Sequence[str],
) -> str:
    """One arm's compose file from the production compose (arm_compose.py)."""
    s = base
    first = s.index("\nname: ") + 1 if "\nname: " in s else 0
    s = "# cache_parity arm %s. Temporary.\n" % name + s[first:]
    s = _sub(s, r"^name: .*$", "name: %s" % name)
    s = _sub(s, r"^  [a-z0-9-]+:\n(?=(?:    #.*\n)*    image: )", "  cmp:\n")
    s = re.sub(r"(  cmp:\n)(?:    #.*\n)+", r"\1", s, count=1)
    if image:
        s = _sub(s, r"^    image: .*$", "    image: " + image)
    s = _sub(s, r"^    container_name: .*$", "    container_name: %s" % container)
    s = _sub(s, r"^    restart: .*$", '    restart: "no"')
    s = _sub(
        s,
        r'^    ports: \[".*"\]$',
        '    ports: ["127.0.0.1:%d:8001"]' % port,
    )
    s = _sub(
        s,
        r"^      - \{type: bind, source: [^,}]+, target: /cache\}$",
        "      - {type: bind, source: %s, target: /cache}" % cache_dir,
    )
    # the production compose mounts a host CUDA toolkit and the entrypoint
    # script; arms run `vllm serve` directly
    s = _sub(
        s,
        r"^      - \{type: bind, source: /usr/local/cuda[^,}]*, target: /usr/local/cuda, read_only: true\}\n",
        "",
        required=False,
    )
    s = _sub(
        s,
        r"^      - \{type: bind, source: [^,}]*entrypoint\.sh, target: /app/entrypoint\.sh, read_only: true\}\n",
        "",
        required=False,
    )
    start = s.index("    environment:\n") + len("    environment:\n")
    end = s.index("    deploy:\n")
    s = (
        s[:start]
        + "".join('      %s: "%s"\n' % kv for kv in env.items())
        + s[end:]
    )
    s = _sub(
        s,
        r"^              device_ids: \[.*\]$",
        "              device_ids: [%s]" % ",".join('"%s"' % g for g in gpus),
    )
    s = _sub(
        s,
        r'^      test: \["CMD", "/app/healthcheck\.sh"\]$',
        '      test: ["CMD-SHELL", "curl -sf http://127.0.0.1:8001/health >/dev/null '
        '|| exit 1"]',
    )
    tail = s.index('    entrypoint: ["/app/entrypoint.sh"]')
    return (
        s[:tail]
        + '    entrypoint: ["vllm", "serve"]\n    command:\n'
        + "".join("      - '%s'\n" % f for f in flags)
    )


# ---------------------------------------------------------------------------
# engines
# ---------------------------------------------------------------------------


class EngineError(RuntimeError):
    pass


def run(cmd: Sequence[str], **kw: Any) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), text=True, capture_output=True, **kw)


class ComposeEngine:
    """docker compose, one generated compose file per arm."""

    def __init__(self, args: argparse.Namespace, base_text: str, lane: dict[str, Any]):
        self.args = args
        self.base = base_text
        self.lane = lane
        self.container = ""
        self.compose_file = ""
        self.base_url = "http://127.0.0.1:%d" % args.port

    def start(self, arm: ArmSpec, env: dict[str, str], cache_dir: str) -> None:
        a = self.args
        self.container = "sx-cp-%s-%s" % (a.tag, arm.name)
        flags = serve_flags(self.lane, a.port, a.drop_flag, a.flag)
        text = compose_from_base(
            self.base,
            name="sx-cp-%s-%s" % (a.tag, arm.name),
            container=self.container,
            image=a.image,
            port=a.port,
            cache_dir=cache_dir,
            gpus=a.gpus.split(","),
            env=env,
            flags=flags,
        )
        self.compose_file = os.path.join(a.workdir, "compose.%s.yaml" % arm.name)
        with open(self.compose_file, "w") as f:
            f.write(text)
        if shutil.which("docker"):
            r = run(["docker", "compose", "-f", self.compose_file, "config", "-q"])
            if r.returncode:
                raise EngineError("compose config failed: %s" % r.stderr.strip()[-400:])
        if a.dry_run:
            return
        r = run(["docker", "compose", "-f", self.compose_file, "up", "-d"])
        if r.returncode:
            raise EngineError("compose up failed: %s" % r.stderr.strip()[-400:])

    def alive(self) -> bool:
        r = run(["docker", "inspect", "-f", "{{.State.Status}}", self.container])
        return r.returncode == 0 and r.stdout.strip() == "running"

    def logs(self) -> str:
        r = run(["docker", "logs", self.container])
        return (r.stdout or "") + (r.stderr or "")

    def stop(self) -> None:
        if self.compose_file:
            run(["docker", "compose", "-f", self.compose_file, "down"])
        wait_gpus_free(self.args.gpus, timeout=240)


class ScriptEngine:
    """A user script: ``script start|stop|logs|alive`` with CP_* in the env.

    CP_ARM, CP_ENV_JSON (path of a JSON {name: value} to export into the
    engine), CP_CACHE_DIR (host dir for /cache), CP_PORT, CP_LANE, CP_WORKDIR.
    ``logs`` prints the engine log, ``alive`` exits 0 while the engine runs.
    """

    def __init__(self, args: argparse.Namespace, lane_name: str):
        self.args = args
        self.lane_name = lane_name
        self.base_url = "http://127.0.0.1:%d" % args.port
        self.arm = ""

    def _env(self) -> dict[str, str]:
        e = dict(os.environ)
        e.update(
            CP_ARM=self.arm,
            CP_PORT=str(self.args.port),
            CP_LANE=self.lane_name,
            CP_WORKDIR=self.args.workdir,
            CP_ENV_JSON=os.path.join(self.args.workdir, "env.%s.json" % self.arm),
            CP_CACHE_DIR=getattr(self, "cache_dir", ""),
        )
        return e

    def _call(self, verb: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [self.args.engine_script, verb],
            text=True,
            capture_output=True,
            env=self._env(),
        )

    def start(self, arm: ArmSpec, env: dict[str, str], cache_dir: str) -> None:
        self.arm = arm.name
        self.cache_dir = cache_dir
        with open(self._env()["CP_ENV_JSON"], "w") as f:
            json.dump(env, f, indent=1, sort_keys=True)
        r = self._call("start")
        if r.returncode:
            raise EngineError("engine script start failed: %s" % r.stderr[-400:])

    def alive(self) -> bool:
        return self._call("alive").returncode == 0

    def logs(self) -> str:
        return self._call("logs").stdout

    def stop(self) -> None:
        self._call("stop")


def wait_gpus_free(gpus: str, timeout: float) -> None:
    """Wait until the GPUs of the previous arm have released their memory."""
    if not shutil.which("nvidia-smi"):
        time.sleep(15)
        return
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                gpus,
            ]
        )
        try:
            used = [int(x) for x in r.stdout.split()]
        except ValueError:
            used = []
        if used and max(used) < 1500:
            return
        time.sleep(5)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def http_json(url: str, payload: Any = None, timeout: float = 900) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def http_text(url: str, timeout: float = 30) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode()


def healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(base_url + "/health", timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_healthy(engine: Any, timeout: float, poll: float = 5.0) -> float:
    """Seconds until /health answers; raises EngineError if the engine dies."""
    t0 = time.time()
    while True:
        if healthy(engine.base_url):
            return time.time() - t0
        if not engine.alive():
            raise EngineError("engine exited before becoming healthy")
        if time.time() - t0 > timeout:
            raise EngineError("engine not healthy after %.0f s" % timeout)
        time.sleep(poll)


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------

_WORDS = (
    "river stone market lantern window copper garden signal bridge harbor "
    "ledger canvas timber orchard compass velvet quartz meadow engine pillar "
    "anchor ribbon thunder marble saddle fabric valley trumpet beacon cobalt "
    "pepper violin glacier hammer jasmine kettle lagoon mosaic nectar oyster "
    "paddle quiver rafter sketch tunnel umbra vessel willow yonder zephyr "
    "archive blanket cistern doorway ember fixture grove hollow island jigsaw"
).split()


class _Rng:
    """Small deterministic LCG so the prompts do not depend on Python's version."""

    def __init__(self, seed: int):
        self.state = (seed * 2654435761 + 12345) & 0xFFFFFFFF

    def next(self, n: int) -> int:
        self.state = (self.state * 1103515245 + 12345) & 0x7FFFFFFF
        return (self.state >> 8) % n


def filler(seed: int, n_chars: int) -> str:
    rng = _Rng(seed)
    out: list[str] = []
    size = 0
    while size < n_chars:
        length = 8 + rng.next(10)
        words = [_WORDS[rng.next(len(_WORDS))] for _ in range(length)]
        sentence = " ".join(words).capitalize() + ". "
        out.append(sentence)
        size += len(sentence)
    return "".join(out)[:n_chars]


def needle_document(seed: int, n_chars: int, needles: int = 8) -> tuple[str, list[str]]:
    rng = _Rng(seed + 7)
    codes = ["%s-%04d" % ("ABCDEFGH"[i], 1000 + rng.next(9000)) for i in range(needles)]
    step = n_chars // (needles + 1)
    parts = []
    for i, code in enumerate(codes):
        parts.append(filler(seed + 100 + i, step))
        parts.append("\nNote %d: the vault code number %d is %s.\n" % (i + 1, i + 1, code))
    parts.append(filler(seed + 999, step))
    return "".join(parts), codes


def code_module(seed: int, n_chars: int) -> tuple[str, str]:
    """A long Python module of small functions; returns (source, answer name)."""
    rng = _Rng(seed + 3)
    lines = ['"""Generated helper module."""', ""]
    target = None
    idx = 0
    while sum(len(x) + 1 for x in lines) < n_chars:
        a, b, m = 2 + rng.next(30), rng.next(50), 50 + rng.next(60)
        name = "f_%04d" % idx
        lines += [
            "def %s(x):" % name,
            '    """%s"""' % filler(seed + idx, 60).strip(),
            "    return (x * %d + %d) %% %d" % (a, b, m),
            "",
        ]
        if target is None and idx > 5 and (5 * a + b) % m == 14:
            target = name
        idx += 1
    if target is None:
        target = "f_0000"
        lines += ["def f_target(x):", "    return (x * 3 + 4) % 10", ""]
        target = "f_target"
    return "\n".join(lines), target


@dataclasses.dataclass
class PromptSpec:
    id: str
    kind: str
    target_tokens: int
    max_tokens: int
    chars: int  # calibrated size of the long part (0 for short prompts)
    system: str = ""
    user: str = ""

    def messages(self) -> list[dict[str, str]]:
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": self.user})
        return msgs


def _session(seed: int) -> str:
    return "Session %s. " % hashlib.sha256(str(seed).encode()).hexdigest()[:12]


def build_user(kind: str, seed: int, chars: int) -> str:
    """User message of one prompt; a session id first avoids shared prefixes."""
    s = _session(seed)
    if kind == "code_lru":
        return s + (
            "Write a complete Python implementation of a thread-safe LRU cache "
            "with per-entry TTL, with type hints and pytest unit tests. Output "
            "only code."
        )
    if kind == "math":
        return s + (
            "Work step by step. A tank has 3 inlet pipes filling it in 6, 8 and "
            "12 hours and one drain emptying it in 10 hours. If all four are "
            "open, how long to fill the tank? Then convert the answer to "
            "minutes and seconds."
        )
    if kind == "zh":
        return s + "请用中文写一段大约三百字的说明：为什么 GPU 上的矩阵乘法比 CPU 快，并举一个例子。"
    if kind == "translate":
        return s + (
            "Translate into French, German and Japanese: 'The compiler caches "
            "its work so that the second start is faster, but the results must "
            "not change.' Give the three translations as a numbered list."
        )
    if kind == "logic":
        return s + (
            "Four friends (Ana, Ben, Cy, Dee) each own a different pet (cat, "
            "dog, fish, bird). Ana does not own the cat or the dog. Ben owns "
            "the bird or the fish. Cy does not own the fish. Dee owns the dog. "
            "Work out who owns which pet, showing the reasoning."
        )
    if kind == "json":
        return s + (
            "Return only a JSON array of 8 objects with the fields id (int), "
            "city (string), population_millions (number) and country_code "
            "(string) for 8 large cities in 8 different countries."
        )
    if kind.startswith("needle"):
        text, _codes = needle_document(seed, chars)
        return (
            s
            + "Read the document and answer at the end.\n\n"
            + text
            + "\n\nList all vault codes in order of their note number, one per line, "
            "as 'number: code'."
        )
    if kind == "summary":
        text = filler(seed, chars)
        return (
            s
            + "Summarize the following report in exactly five bullet points.\n\n"
            + text
        )
    if kind == "code_long":
        source, answer = code_module(seed, chars)
        return (
            s
            + "Here is a Python module.\n\n```python\n"
            + source
            + "\n```\n\nWhich function returns 14 for x = 5? Then write a short "
            "docstring for f_0003. Answer concisely."
        )
    raise KeyError(kind)


def prompt_plan(long_lengths: Sequence[int]) -> list[PromptSpec]:
    """Six short prompts and one long prompt per length, plus one more needle."""
    shorts = [
        ("code_lru", 512),
        ("math", 256),
        ("zh", 384),
        ("json", 256),
        ("translate", 256),
        ("logic", 384),
    ]
    longs = []
    kinds = ["needle", "summary", "code_long", "needle", "needle"]
    for i, length in enumerate(long_lengths):
        longs.append((kinds[i % len(kinds)], length))
    mid = long_lengths[len(long_lengths) // 2]
    longs.append(("needle", mid + 1000))
    plan = []
    for kind, max_tokens in shorts:
        plan.append(PromptSpec("", kind, 0, max_tokens, 0))
    for kind, length in longs:
        plan.append(PromptSpec("", kind, length, 256 if kind == "needle" else 512, 0))
    return [
        dataclasses.replace(
            p,
            id="p%02d_%s%s"
            % (i + 1, p.kind, "_%dk" % (p.target_tokens // 1000) if p.target_tokens else ""),
        )
        for i, p in enumerate(plan)
    ]


def calibrate_prompts(
    plan: list[PromptSpec],
    count_tokens: Callable[[str], int] | None,
    seed: int = 20261002,
) -> list[PromptSpec]:
    """Fix the long parts to ~target tokens (engine /tokenize, 4 chars/token else)."""
    out = []
    for i, p in enumerate(plan):
        pseed = seed + i
        if p.target_tokens == 0:
            out.append(dataclasses.replace(p, user=build_user(p.kind, pseed, 0)))
            continue
        chars = p.target_tokens * 4
        user = build_user(p.kind, pseed, chars)
        if count_tokens is not None:
            for _ in range(5):
                n = count_tokens(user)
                if abs(n - p.target_tokens) <= max(16, p.target_tokens // 100):
                    break
                chars = max(1000, int(chars * p.target_tokens / max(n, 1)))
                user = build_user(p.kind, pseed, chars)
        out.append(dataclasses.replace(p, chars=chars, user=user))
    return out


def save_prompts(path: str, prompts: Sequence[PromptSpec]) -> None:
    with open(path, "w") as f:
        json.dump([dataclasses.asdict(p) for p in prompts], f, indent=1)


def load_prompts(path: str) -> list[PromptSpec]:
    with open(path) as f:
        return [PromptSpec(**d) for d in json.load(f)]


# ---------------------------------------------------------------------------
# requests and metrics
# ---------------------------------------------------------------------------


def chat_request(base_url: str, model: str, p: PromptSpec, timeout: float) -> dict:
    payload = {
        "model": model,
        "messages": p.messages(),
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "max_tokens": p.max_tokens,
        "stream": False,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    try:
        r = http_json(base_url + "/v1/chat/completions", payload, timeout)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return {"id": p.id, "ok": False, "error": repr(exc), "seconds": time.time() - t0}
    choice = r["choices"][0]
    return {
        "id": p.id,
        "ok": True,
        "finish_reason": choice.get("finish_reason"),
        "token_ids": choice.get("token_ids"),
        "text": (choice.get("message") or {}).get("content"),
        "completion_tokens": (r.get("usage") or {}).get("completion_tokens"),
        "prompt_tokens": (r.get("usage") or {}).get("prompt_tokens"),
        "seconds": time.time() - t0,
    }


_METRIC = re.compile(
    r'^(vllm:spec_decode_[a-z_]+?)(?:_total)?(\{[^}]*\})?\s+([0-9.eE+-]+)\s*$'
)


def parse_spec_metrics(text: str) -> dict[str, float]:
    """Spec-decode counters of /metrics: drafts, draft tokens, accepted, per position."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _METRIC.match(line)
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2) or "", float(m.group(3))
        key = name.replace("vllm:spec_decode_", "")
        pos = re.search(r'position="(\d+)"', labels)
        if pos:
            key = "%s[%s]" % (key, pos.group(1))
        out[key] = out.get(key, 0.0) + value
    return out


def spec_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    keys = sorted(set(before) | set(after))
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in keys}


# ---------------------------------------------------------------------------
# engine log parsing
# ---------------------------------------------------------------------------

_PREFIX = re.compile(r"^\((?P<who>[A-Za-z_]+\d*)(?: pid=\d+)?\)\s*")
_FLOAT = r"(?P<v>[0-9]+(?:\.[0-9]+)?)"
_SUM_PATTERNS: dict[str, re.Pattern[str]] = {
    "torch_compile_s": re.compile(r"torch\.compile took " + _FLOAT + r" s in total"),
    "torch_compile_warmup_s": re.compile(
        r"torch\.compile and initial profiling/warmup run together took "
        + _FLOAT
        + r" s in total"
    ),
    "dynamo_s": re.compile(r"Dynamo bytecode transform time: " + _FLOAT + r" s"),
    "compile_range_s": re.compile(
        r"Compiling a graph for compile range .* takes " + _FLOAT + r" s"
    ),
    "cache_load_s": re.compile(
        r"Directly load the compiled graph\(s\) for compile range .* took "
        + _FLOAT
        + r" s"
    ),
    "graph_capture_s": re.compile(r"Graph capturing finished in " + _FLOAT + r" secs"),
    "model_loading_s": re.compile(r"Model loading took .* and " + _FLOAT + r" seconds"),
    "init_engine_s": re.compile(
        r"init engine \(profile, create kv cache, warmup model\) took " + _FLOAT + r" s"
    ),
    "profiling_warmup_s": re.compile(r"Initial profiling/warmup run took " + _FLOAT + r" s"),
}
_COUNT_PATTERNS: dict[str, str] = {
    "aot_loaded": "Directly load AOT compilation from path",
    "aot_saved": "saved AOT compiled function to",
    "graphs_cached": "Cache the graph of compile range",
    "cache_disabled_notice": "vLLM's torch.compile cache is disabled",
    "cache_dir_notice": "for vLLM's torch.compile",
}
_ERROR_LEVEL = re.compile(r"\s*ERROR\s")
_COUNTERS = re.compile(r"SX compile-cache counters: (?P<kv>.*)$")
_KV = re.compile(r"(\w+)=(-?\d+)")
_KV_CACHE = re.compile(r"GPU KV cache size: ([0-9,]+) tokens")
_IDENTITY = re.compile(r"sx-compile-cache build identity (\w+)")


def parse_engine_log(text: str) -> dict[str, Any]:
    """Phase times, reuse counts, counters and error lines of one engine log.

    ``<phase>`` lists hold every value of the log; ``by_rank[<who>][<phase>]``
    the same per process (``Worker_TP0``, ``EngineCore``, ``main``).
    ``bad_lines`` are the patterns that fail a run (BAD_LOG_PATTERNS),
    ``errors`` any other ERROR or Traceback line (reported, not failing).
    """
    out: dict[str, Any] = {k: [] for k in _SUM_PATTERNS}
    out.update({k: 0 for k in _COUNT_PATTERNS})
    out["by_rank"] = {}
    out["counters"] = {}
    out["bad_lines"] = []
    out["errors"] = []
    out["identity"] = None
    out["kv_cache_tokens"] = None
    for raw in text.splitlines():
        m = _PREFIX.match(raw)
        who = m.group("who") if m else "main"
        line = raw[m.end() :] if m else raw
        for key, pat in _SUM_PATTERNS.items():
            hit = pat.search(line)
            if hit:
                value = float(hit.group("v"))
                out[key].append(value)
                out["by_rank"].setdefault(who, {}).setdefault(key, []).append(value)
        for key, needle in _COUNT_PATTERNS.items():
            if needle in line:
                out[key] += 1
        hit = _COUNTERS.search(line)
        if hit:
            out["counters"][who] = {k: int(v) for k, v in _KV.findall(hit.group("kv"))}
        hit = _KV_CACHE.search(line)
        if hit and out["kv_cache_tokens"] is None:
            out["kv_cache_tokens"] = int(hit.group(1).replace(",", ""))
        hit = _IDENTITY.search(line)
        if hit and out["identity"] is None:
            out["identity"] = hit.group(1)
        if any(p in line for p in BAD_LOG_PATTERNS):
            out["bad_lines"].append(line.strip()[:300])
        elif "Traceback (most recent call last)" in line or _ERROR_LEVEL.match(line):
            out["errors"].append(line.strip()[:300])
    out["bad_lines"] = list(dict.fromkeys(out["bad_lines"]))
    out["errors"] = list(dict.fromkeys(out["errors"]))
    return out


def phase_summary(parsed: dict[str, Any]) -> dict[str, float]:
    """Seconds per phase; per-rank phases are summed per rank, then the slowest rank."""

    def mx(key: str) -> float:
        return max(parsed[key]) if parsed[key] else 0.0

    per_rank_compile = [
        sum(r.get("torch_compile_s", [])) + sum(r.get("torch_compile_warmup_s", []))
        for r in parsed["by_rank"].values()
    ]
    return {
        "model_loading_s": mx("model_loading_s"),
        "torch_compile_total_s": max(per_rank_compile) if per_rank_compile else 0.0,
        "dynamo_max_s": mx("dynamo_s"),
        "compile_range_max_s": mx("compile_range_s"),
        "cache_load_max_s": mx("cache_load_s"),
        "graph_capture_max_s": mx("graph_capture_s"),
        "init_engine_max_s": mx("init_engine_s"),
    }


def reuse_stats(parsed: dict[str, Any]) -> dict[str, int]:
    """Sum of the per-rank counters (0 for ranks that did not log one)."""
    total: dict[str, int] = {}
    for counters in parsed["counters"].values():
        for k, v in counters.items():
            total[k] = total.get(k, 0) + v
    return total


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class PairResult:
    a: str
    b: str
    equal: int = 0
    total: int = 0
    differences: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    spec_equal: bool | None = None
    spec_difference: dict[str, tuple[float, float]] = dataclasses.field(default_factory=dict)

    @property
    def identical(self) -> bool:
        return self.equal == self.total and self.spec_equal is not False and self.total > 0


def _first_diff(x: Sequence[int], y: Sequence[int]) -> int | None:
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            return i
    return None if len(x) == len(y) else min(len(x), len(y))


def compare_pair(a_name: str, a: dict, b_name: str, b: dict) -> PairResult:
    res = PairResult(a_name, b_name)
    by_id = {r["id"]: r for r in b.get("outputs", [])}
    for ra in a.get("outputs", []):
        rb = by_id.get(ra["id"])
        res.total += 1
        if rb is None or not ra.get("ok") or not rb.get("ok"):
            res.differences.append(
                {"id": ra["id"], "reason": "request failed or missing in one arm"}
            )
            continue
        ta, tb = ra.get("token_ids"), rb.get("token_ids")
        if ta is None or tb is None:
            # no token ids from the API: fall back to the text
            ta, tb = list(ra.get("text") or ""), list(rb.get("text") or "")
            unit = "char"
        else:
            unit = "token"
        pos = _first_diff(ta, tb)
        if pos is None:
            res.equal += 1
        else:
            res.differences.append(
                {
                    "id": ra["id"],
                    "reason": "first difference at %s %d of %d/%d"
                    % (unit, pos, len(ta), len(tb)),
                    "position": pos,
                    "lengths": [len(ta), len(tb)],
                    "a_tail": (ra.get("text") or "")[max(0, pos * 3 - 40) : pos * 3 + 60],
                    "b_tail": (rb.get("text") or "")[max(0, pos * 3 - 40) : pos * 3 + 60],
                }
            )
    sa, sb = a.get("spec"), b.get("spec")
    if sa is not None and sb is not None:
        keys = sorted(set(sa) | set(sb))
        diff = {k: (sa.get(k, 0.0), sb.get(k, 0.0)) for k in keys if sa.get(k, 0.0) != sb.get(k, 0.0)}
        res.spec_equal = not diff
        res.spec_difference = diff
    return res


def reuse_verdict(cold: dict, warm: dict, mode: str) -> list[str]:
    """Reasons the warm arm did not reuse what the cold arm saved (empty = ok)."""
    problems: list[str] = []
    cs, ws = cold.get("reuse") or {}, warm.get("reuse") or {}
    if not cs or not ws:
        return ["no 'SX compile-cache counters' line in the cold or warm log"]
    if mode == "aot":
        saved = cs.get("num_aot_artifacts_saved", 0)
        if saved <= 0:
            problems.append("cold arm saved no AOT artifact")
        if ws.get("num_aot_artifacts_loaded", 0) < saved:
            problems.append(
                "warm arm loaded %d AOT artifacts, cold saved %d"
                % (ws.get("num_aot_artifacts_loaded", 0), saved)
            )
        if ws.get("num_aot_compiles", 0) != 0:
            problems.append("warm arm compiled %d AOT graphs" % ws["num_aot_compiles"])
    else:
        saved = cs.get("num_compiled_artifacts_saved", 0)
        if saved <= 0:
            problems.append("cold arm saved no compiled artifact")
        if ws.get("num_compiled_artifacts_loaded", 0) <= 0:
            problems.append("warm arm loaded no compiled artifact")
        if ws.get("num_compiled_artifacts_saved", 0) != 0:
            problems.append(
                "warm arm compiled and saved %d artifacts"
                % ws["num_compiled_artifacts_saved"]
            )
    return problems


def build_report(
    results: dict[str, dict], mode: str, lane: str
) -> tuple[dict[str, Any], int]:
    """Verdict, pair table and exit code from the saved arm results."""
    reasons: list[str] = []
    notes: list[str] = []
    pairs: list[PairResult] = []
    for a, b in GATE_PAIRS:
        if a in results and b in results:
            res = compare_pair(a, results[a], b, results[b])
            pairs.append(res)
            if not res.identical:
                reasons.append(
                    "%s vs %s: %d/%d prompts identical%s"
                    % (
                        a,
                        b,
                        res.equal,
                        res.total,
                        "" if res.spec_equal is not False else ", draft counters differ",
                    )
                )
    controls: list[PairResult] = []
    for a, b in CONTROL_PAIRS:
        if a in results and b in results:
            controls.append(compare_pair(a, results[a], b, results[b]))
    for c in controls:
        if not c.identical:
            notes.append(
                "control %s vs %s differs (%d/%d identical): the reference itself is "
                "not reproducible across restarts or between AOT and non-AOT; a "
                "cache pair difference is then not evidence against the cache"
                % (c.a, c.b, c.equal, c.total)
            )
    for name, r in results.items():
        if r.get("failed"):
            reasons.append("arm %s failed: %s" % (name, r["failed"]))
        if name in ("cold", "warm", "warm2") and r.get("log", {}).get("bad_lines"):
            for line in r["log"]["bad_lines"][:5]:
                reasons.append("arm %s log: %s" % (name, line))
        if r.get("log", {}).get("errors"):
            notes.append(
                "arm %s log has %d ERROR/Traceback line(s), e.g. %s"
                % (name, len(r["log"]["errors"]), r["log"]["errors"][0][:160])
            )
        for o in r.get("outputs", []):
            if not o.get("ok"):
                reasons.append("arm %s request %s failed: %s" % (name, o["id"], o.get("error")))
    if "cold" in results and "warm" in results:
        reasons += ["reuse: " + p for p in reuse_verdict(results["cold"], results["warm"], mode)]
    if "inval" in results and not results["inval"].get("failed"):
        iv = results["inval"].get("reuse") or {}
        recompiled = (
            iv.get("num_compiled_artifacts_saved", 0) > 0
            or iv.get("num_aot_compiles", 0) > 0
        )
        if not iv:
            reasons.append("invalidation: no counters line in the inval log")
        elif not recompiled:
            reasons.append(
                "invalidation: a changed VLLM_ switch reused the cache (the key does "
                "not cover it)"
            )
    timing = {
        n: {
            "healthy_s": r.get("healthy_s"),
            **(r.get("phases") or {}),
        }
        for n, r in results.items()
    }
    report = {
        "lane": lane,
        "mode": mode,
        "verdict": "FAIL" if reasons else "PASS",
        "reasons": reasons,
        "notes": notes,
        "pairs": [dataclasses.asdict(p) | {"identical": p.identical} for p in pairs],
        "controls": [dataclasses.asdict(p) | {"identical": p.identical} for p in controls],
        "timing": timing,
    }
    return report, (1 if reasons else 0)


def format_report(report: dict[str, Any]) -> str:
    lines = [
        "cache_parity lane=%s mode=%s: %s" % (report["lane"], report["mode"], report["verdict"]),
        "",
        "| arm | healthy s | model load | compile (rank) | dynamo | cache load | graph capture |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, t in report["timing"].items():
        lines.append(
            "| %s | %s | %s | %s | %s | %s | %s |"
            % (
                arm,
                _f(t.get("healthy_s")),
                _f(t.get("model_loading_s")),
                _f(t.get("torch_compile_total_s")),
                _f(t.get("dynamo_max_s")),
                _f(t.get("cache_load_max_s")),
                _f(t.get("graph_capture_max_s")),
            )
        )
    lines += ["", "| pair | identical prompts | draft counters | verdict |", "|---|---:|---|---|"]
    for p in report["pairs"] + report["controls"]:
        spec = "-" if p["spec_equal"] is None else ("equal" if p["spec_equal"] else "DIFFER")
        lines.append(
            "| %s vs %s | %d/%d | %s | %s |"
            % (p["a"], p["b"], p["equal"], p["total"], spec, "ok" if p["identical"] else "DIFFERENT")
        )
    for p in report["pairs"] + report["controls"]:
        for d in p["differences"][:6]:
            lines.append("  %s vs %s %s: %s" % (p["a"], p["b"], d["id"], d["reason"]))
        for k, (x, y) in list(p["spec_difference"].items())[:6]:
            lines.append("  %s vs %s %s: %s != %s" % (p["a"], p["b"], k, x, y))
    for n in report["notes"]:
        lines.append("NOTE: " + n)
    if report["reasons"]:
        lines += ["", "FAILED:"] + ["  - " + r for r in report["reasons"]]
    return "\n".join(lines)


def _f(x: Any) -> str:
    return "-" if x is None else ("%.1f" % x)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def arm_env(lane: dict[str, Any], arm: ArmSpec, user_env: dict[str, str]) -> dict[str, str]:
    env = {**COMMON_ENV, **lane["env"], **user_env}
    env.update(arm.extra_env)
    env["SX_OPT_COMPILE_CACHE"] = arm.switch
    return env


def wipe_dir(path: str, image: str | None, log: Callable[[str], None]) -> None:
    """Remove a cache dir whose files the engine container created (other uid)."""
    try:
        shutil.rmtree(path)
        return
    except OSError as exc:
        log("rmtree %s failed (%s); trying sudo, then a root container" % (path, exc))
    if shutil.which("sudo") and run(["sudo", "-n", "rm", "-rf", path]).returncode == 0:
        if not os.path.exists(path):
            return
    if image and shutil.which("docker"):
        run(
            [
                "docker", "run", "--rm", "--user", "0", "--entrypoint", "sh",
                "-v", "%s:/w" % path, image, "-c",
                "rm -rf /w/* /w/.[!.]* /w/..?* 2>/dev/null; true",
            ]
        )
        shutil.rmtree(path, ignore_errors=True)
    if os.path.exists(path):
        raise EngineError("cannot remove %s: remove it by hand (sudo rm -rf)" % path)


def prepare_cache_dir(
    root: str,
    arm: ArmSpec,
    seed: str | None,
    log: Callable[[str], None],
    image: str | None = None,
) -> str:
    path = os.path.join(root, "cache-" + arm.cache)
    if arm.fresh and os.path.isdir(path):
        wipe_dir(path, image, log)
    if not os.path.isdir(path):
        os.makedirs(path)
        if seed:
            log("seeding %s from %s" % (path, seed))
            shutil.copytree(seed, path, dirs_exist_ok=True, symlinks=True)
        # the container user is not the operator
        for dirpath, dirnames, filenames in os.walk(path):
            for n in dirnames + filenames:
                try:
                    os.chmod(os.path.join(dirpath, n), 0o777)
                except OSError:
                    pass
    os.chmod(path, 0o777)
    return path


def run_arm(
    engine: Any,
    arm: ArmSpec,
    lane: dict[str, Any],
    args: argparse.Namespace,
    prompts: list[PromptSpec] | None,
    log: Callable[[str], None],
) -> tuple[dict[str, Any], list[PromptSpec] | None]:
    env = arm_env(lane, arm, dict(kv.split("=", 1) for kv in args.env))
    result: dict[str, Any] = {"arm": arm.name, "switch": arm.switch}
    t0 = time.time()
    try:
        cache_dir = prepare_cache_dir(
            args.cache_root, arm, args.cache_seed, log, getattr(args, "image", None)
        )
        result["cache_dir"] = cache_dir
        log("[%s] start (SX_OPT_COMPILE_CACHE=%s, cache %s)" % (arm.name, arm.switch, cache_dir))
        engine.start(arm, env, cache_dir)
        if args.dry_run:
            return result, prompts
        result["healthy_s"] = wait_healthy(engine, args.health_timeout)
        log("[%s] healthy after %.0f s" % (arm.name, result["healthy_s"]))
        if prompts is None:
            prompts = load_or_make_prompts(engine, lane, args, log)
        before = parse_spec_metrics(http_text(engine.base_url + "/metrics"))
        outputs = []
        for p in prompts:
            out = chat_request(engine.base_url, SERVED_NAME, p, args.request_timeout)
            log(
                "[%s] %s: %s, %s tokens, %.1f s"
                % (arm.name, p.id, "ok" if out["ok"] else "FAILED", out.get("completion_tokens"), out["seconds"])
            )
            outputs.append(out)
        after = parse_spec_metrics(http_text(engine.base_url + "/metrics"))
        result["outputs"] = outputs
        if lane["mtp"] or any(k.startswith("num_drafts") for k in after):
            result["spec"] = spec_delta(before, after)
    except EngineError as exc:
        result["failed"] = str(exc)
        log("[%s] FAILED: %s" % (arm.name, exc))
    finally:
        text = ""
        try:
            text = engine.logs()
        except Exception as exc:  # noqa: BLE001
            result.setdefault("failed", "could not read the engine log: %r" % exc)
        with open(os.path.join(args.workdir, "engine_%s.log" % arm.name), "w") as f:
            f.write(text)
        parsed = parse_engine_log(text)
        result["log"] = parsed
        result["phases"] = phase_summary(parsed)
        result["reuse"] = reuse_stats(parsed)
        result["wall_s"] = time.time() - t0
        if not args.dry_run:
            engine.stop()
    return result, prompts


def load_or_make_prompts(engine: Any, lane: dict[str, Any], args: argparse.Namespace, log: Callable[[str], None]) -> list[PromptSpec]:
    path = args.prompts_json or os.path.join(args.workdir, "prompts.json")
    if os.path.exists(path):
        return load_prompts(path)

    def count(text: str) -> int:
        r = http_json(engine.base_url + "/tokenize", {"model": SERVED_NAME, "prompt": text}, 300)
        return int(r["count"])

    plan = prompt_plan(lane["long_lengths"])
    prompts = calibrate_prompts(plan, count)
    save_prompts(path, prompts)
    log("prompts written to %s (%d prompts)" % (path, len(prompts)))
    return prompts


def make_engine(args: argparse.Namespace, lane_name: str, lane: dict[str, Any]) -> Any:
    if args.engine_script:
        return ScriptEngine(args, lane_name)
    with open(args.compose) as f:
        base = f.read()
    return ComposeEngine(args, base, lane)


def cmd_run(args: argparse.Namespace, engine: Any | None = None) -> int:
    lane = LANES[args.lane]
    os.makedirs(args.workdir, exist_ok=True)
    os.makedirs(args.cache_root, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    table = arm_table(args.cache_mode)
    unknown = [a for a in arms if a not in table]
    if unknown:
        print("unknown arm(s) %s; choose from %s" % (unknown, sorted(table)), file=sys.stderr)
        return 2
    if "warm" in arms and "cold" not in arms and not os.path.isdir(os.path.join(args.cache_root, "cache-on")):
        print("arm warm needs cold (or an existing cache-on directory)", file=sys.stderr)
        return 2
    engine = engine or make_engine(args, args.lane, lane)

    def log(msg: str) -> None:
        print("%s %s" % (time.strftime("%H:%M:%S"), msg), flush=True)

    if args.stop_first:
        subprocess.run(args.stop_first, shell=True, check=False)
    results: dict[str, dict] = {}
    prompts: list[PromptSpec] | None = None
    try:
        if args.prompts_json and os.path.exists(args.prompts_json):
            prompts = load_prompts(args.prompts_json)
        for name in arms:
            res, prompts = run_arm(engine, table[name], lane, args, prompts, log)
            results[name] = res
            with open(os.path.join(args.workdir, "arm_%s.json" % name), "w") as f:
                json.dump(res, f, indent=1)
            if res.get("failed") and name in ("off", "cold"):
                log("arm %s failed; later arms depend on it, stopping" % name)
                break
    finally:
        if args.restart_after:
            subprocess.run(args.restart_after, shell=True, check=False)
    if args.dry_run:
        return 0
    report, code = build_report(results, args.cache_mode, args.lane)
    with open(os.path.join(args.workdir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    text = format_report(report)
    with open(os.path.join(args.workdir, "report.md"), "w") as f:
        f.write(text + "\n")
    print()
    print(text)
    if code:
        print("\n*** CACHE PARITY FAILED: do not enable SX_OPT_COMPILE_CACHE=%s in production ***" % args.cache_mode, file=sys.stderr)
    return code


def cmd_compare(args: argparse.Namespace) -> int:
    results = {}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        with open(path) as f:
            results[name] = json.load(f)
    report, code = build_report(results, args.cache_mode, args.lane)
    print(format_report(report))
    return code


def cmd_prompts(args: argparse.Namespace) -> int:
    lane = LANES[args.lane]
    plan = prompt_plan(lane["long_lengths"])
    count = None
    if args.tokenize_port:

        def count(text: str) -> int:  # noqa: F811
            r = http_json(
                "http://127.0.0.1:%d/tokenize" % args.tokenize_port,
                {"model": SERVED_NAME, "prompt": text},
                300,
            )
            return int(r["count"])

    prompts = calibrate_prompts(plan, count)
    save_prompts(args.out, prompts)
    for p in prompts:
        print("%-22s target %6d tokens, %5d chars, max_tokens %d" % (p.id, p.target_tokens, len(p.user), p.max_tokens))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="start the arms one after the other and compare")
    r.add_argument("--lane", choices=sorted(LANES), default="nomtp")
    r.add_argument("--cache-mode", choices=["1", "subgraph", "aot"], default="1",
                   help="SX_OPT_COMPILE_CACHE value of the cache-on arms")
    r.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                   help="comma list of off,cold,warm,off2,off_noaot,warm2,inval (default off,cold,warm)")
    r.add_argument("--compose", default="/opt/shixiang-inference/docker-dflash2/compose.swift15-flashnext-tp4-gpu0123.yaml")
    r.add_argument("--image", default="shixiang/1cat-vllm-v100:heavily-modified-v1-mtp2-sm70main")
    r.add_argument("--engine-script", help="launcher script instead of docker compose (see README)")
    r.add_argument("--gpus", default="4,5,6,7")
    r.add_argument("--port", type=int, default=8141)
    r.add_argument("--tag", default="cp")
    r.add_argument("--workdir", default="/mnt/2t/build/cache_parity/run")
    r.add_argument("--cache-root", default="/mnt/2t/build/cache_parity/caches",
                   help="holds cache-off/ and cache-on/, each bound to /cache")
    r.add_argument("--cache-seed", help="directory copied into every fresh cache dir (e.g. a copy of the production /cache without vllm/)")
    r.add_argument("--env", action="append", default=[], metavar="K=V", help="extra engine env, every arm")
    r.add_argument("--flag", action="append", default=[], help="extra `vllm serve` flag")
    r.add_argument("--drop-flag", action="append", default=[], help="drop default flags starting with this")
    r.add_argument("--prompts-json", help="prompt set to use (created on the first arm if missing)")
    r.add_argument("--health-timeout", type=float, default=3000)
    r.add_argument("--request-timeout", type=float, default=900)
    r.add_argument("--stop-first", help="shell command run before the first arm (stop the instance on these GPUs)")
    r.add_argument("--restart-after", help="shell command run after the last arm")
    r.add_argument("--dry-run", action="store_true", help="write the compose files only")
    r.set_defaults(fn=cmd_run)

    c = sub.add_parser("compare", help="compare saved arm_*.json files")
    c.add_argument("--arm", action="append", required=True, metavar="NAME=PATH")
    c.add_argument("--lane", default="nomtp")
    c.add_argument("--cache-mode", default="1")
    c.set_defaults(fn=cmd_compare)

    p = sub.add_parser("prompts", help="write the fixed prompt set")
    p.add_argument("--out", required=True)
    p.add_argument("--lane", choices=sorted(LANES), default="nomtp")
    p.add_argument("--tokenize-port", type=int, help="calibrate lengths with this engine's /tokenize")
    p.set_defaults(fn=cmd_prompts)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run" and args.cache_mode == "1":
        args.cache_mode = "subgraph"
    if args.cmd == "compare" and args.cache_mode == "1":
        args.cache_mode = "subgraph"
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
