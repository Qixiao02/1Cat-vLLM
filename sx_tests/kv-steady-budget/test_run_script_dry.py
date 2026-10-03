# SPDX-License-Identifier: Apache-2.0
"""``run_on_v100.sh`` end to end with a fake docker, nvidia-smi and engine.

Run (needs bash, curl and python3 on PATH; skipped otherwise):

    python -m pytest -q sx_tests/kv-steady-budget/test_run_script_dry.py

No GPU and no docker: ``fakebin/docker`` accepts the compose commands the script
issues (``compose config -q`` validates that the generated file exists and
mentions the image), reports the container as running and prints a canned engine
log; ``fakebin/nvidia-smi`` reads the used memory from a file that the fake
engine (a small HTTP server in this process) raises while a request is in
flight. The script runs for real: option parsing, compose generation, the
sampler, the health wait, the stress client, the analysis and the summary.

Asserted: two trials (baseline with the switch off at the first util, then the
switch on) leave their compose files, samples, events, engine logs and result
JSON behind; the compose file of each carries the switch and the util; the
summary table lists both and the script exits 0 when the fake engine stays below
the limit and 1 when it does not; bad usage exits 2.
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import stat
import subprocess
import sys
import threading
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_bash() -> str | None:
    """bash, and on Windows Git for Windows' (the ``bash`` on PATH may be WSL's)."""
    if sys.platform == "win32":
        for root in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
            if root:
                candidate = os.path.join(root, "Git", "bin", "bash.exe")
                if os.path.exists(candidate):
                    return candidate
        return None
    return shutil.which("bash")


BASH = _find_bash()
SCRIPT = os.path.join(HERE, "run_on_v100.sh").replace("\\", "/")

pytestmark = pytest.mark.skipif(
    BASH is None or shutil.which("curl") is None,
    reason="needs bash and curl",
)

ENGINE_LOG = """\
INFO 10-02 12:00:00 [kv_cache_utils.py:1] GPU KV cache size: 97,000 tokens
INFO 10-02 12:00:00 [kv_cache_utils.py:2] Maximum concurrency for 32,768 tokens per request: 2.96x
(Worker_TP0 pid=11) INFO [gpu_worker.py:1] Available KV cache memory: 2.19 GiB
(Worker_TP0 pid=11) INFO Graph capturing finished in 126 secs, took 1.04 GiB
(Worker_TP0 pid=11) INFO KV steady budget [kv=2295000000 total=34089730048 requested=1 free_after_profile=2 activation=3 graph_reserve=4 post_sizing=5 headroom=6 limiting=physical]
(Worker_TP0 pid=11) INFO KV steady audit: OK, 100 MiB above the headroom. Setting SX_OPT_KV_STEADY_RESERVE_MIB=2900 would size the KV cache exactly to the measurement.
"""

FAKE_DOCKER = """#!/usr/bin/env bash
# fake docker for the dry run
if [ "$1" = compose ]; then
  f=""; sub=""
  shift
  while [ $# -gt 0 ]; do
    case "$1" in
      -f) f=$2; shift 2 ;;
      config|up|down) sub=$1; shift ;;
      *) shift ;;
    esac
  done
  case "$sub" in
    config) [ -s "$f" ] && grep -q "image:" "$f" ;;
    up) echo "$f" >> "$FAKE_STATE/up.log"; echo 31499 > "$FAKE_STATE/used"; echo "Started" ;;
    down) echo 5 > "$FAKE_STATE/used"; echo "Removed" ;;
  esac
  exit $?
fi
case "$1" in
  inspect) echo running ;;
  logs) cat "$FAKE_STATE/engine.log" ;;
esac
"""

FAKE_SMI = """#!/usr/bin/env bash
used=$(cat "$FAKE_STATE/used")
case "$*" in
  *index,memory.used,memory.total*)
    for g in 4 5 6 7; do echo "$g, $used, 32768"; done ;;
  *)
    for g in 4 5 6 7; do echo "$used"; done ;;
esac
"""


class Engine(http.server.BaseHTTPRequestHandler):
    state_dir = ""
    peak = 31811

    def log_message(self, *args):
        pass

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            payload = {"count": len(body["prompt"].split()) + 1}
        else:
            with open(os.path.join(self.state_dir, "used"), "w") as handle:
                handle.write(str(self.peak))
            time.sleep(0.2)
            payload = {"usage": {"prompt_tokens": 1, "completion_tokens": body["max_tokens"]}}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def write_exe(path, text):
    with open(path, "w", newline="\n") as handle:
        handle.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


@pytest.fixture
def rig(tmp_path):
    state = tmp_path / "state"
    bindir = tmp_path / "fakebin"
    models = tmp_path / "models"
    for d in (state, bindir, models):
        d.mkdir()
    (state / "used").write_text("5")
    (state / "engine.log").write_text(ENGINE_LOG)
    write_exe(bindir / "docker", FAKE_DOCKER)
    write_exe(bindir / "nvidia-smi", FAKE_SMI)
    # python3 is the interpreter running the test (on Windows `python3` can be
    # the Microsoft Store stub).
    interpreter = sys.executable.replace("\\", "/")
    write_exe(bindir / "python3", f'#!/usr/bin/env bash\nexec "{interpreter}" "$@"\n')
    Engine.state_dir = str(state)
    Engine.peak = 31811
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Engine)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = dict(os.environ)
    env["PATH"] = f"{bindir}{os.pathsep}{env['PATH']}"
    env["FAKE_STATE"] = str(state)
    yield {
        "tmp": tmp_path, "env": env, "models": models, "port": server.server_address[1],
        "state": state,
    }
    server.shutdown()


def run_script(rig, *extra, utils="0.93"):
    out = rig["tmp"] / "out"
    cmd = [
        BASH, SCRIPT, "--image", "img:test", "--models-dir", str(rig["models"]),
        "--cache-dir", str(rig["tmp"] / "cache"), "--lane", "MTP", "--utils", utils,
        "--out", str(out), "--port", str(rig["port"]), "--sample-interval", "1",
        "--start-timeout", "60", *extra,
    ]
    proc = subprocess.run(cmd, env=rig["env"], capture_output=True, text=True, timeout=300)
    return proc, out


def test_both_trials_run_and_the_summary_lists_them(rig):
    proc, out = run_script(rig)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    for tag in ("mtp-s0-u093", "mtp-s1-u093"):
        for name in ("compose.yaml", "samples.csv", "events.log", "engine.log", "stress.json"):
            assert (out / tag / name).exists(), (tag, name)
        assert (out / f"result.{tag}.json").exists()
    c0 = (out / "mtp-s0-u093" / "compose.yaml").read_text()
    c1 = (out / "mtp-s1-u093" / "compose.yaml").read_text()
    assert 'SX_OPT_KV_STEADY_BUDGET: "0"' in c0 and 'SX_OPT_KV_STEADY_BUDGET: "1"' in c1
    assert '"--gpu-memory-utilization=0.93"' in c1
    result = json.loads((out / "result.mtp-s1-u093.json").read_text())
    assert result["pass"] and result["log"]["kv_tokens"] == 97000
    assert result["memory"]["idle_mib"] == 31499
    assert result["memory"]["stress_peak_mib"] == 31811
    summary = (out / "summary.txt").read_text()
    assert "PASS" in summary and summary.count("97000") == 2
    assert "ok (2900)" in summary


def test_switch_one_only_skips_the_baseline(rig):
    proc, out = run_script(rig, "--switch", "1", "--no-stress")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert not (out / "mtp-s0-u093").exists()
    assert (out / "mtp-s1-u093" / "engine.log").exists()
    assert not (out / "mtp-s1-u093" / "stress.json").exists()


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_patch_repo_overlays_the_files_that_differ(rig):
    repo = rig["tmp"] / "repo"
    (repo / "vllm" / "v1").mkdir(parents=True)
    (repo / "vllm" / "v1" / "same.py").write_text("same\n")
    (repo / "vllm" / "v1" / "changed.py").write_text("old\n")

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                       env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                            "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})

    git("init", "-q")
    git("config", "core.autocrlf", "false")
    git("add", "-A")
    git("commit", "-q", "-m", "base")
    git("tag", "base")
    (repo / "vllm" / "v1" / "changed.py").write_text("new\n")
    (repo / "vllm" / "v1" / "added.py").write_text("added\n")
    git("add", "-A")
    git("commit", "-q", "-m", "change")

    proc, out = run_script(
        rig, "--switch", "1", "--no-stress",
        "--patch-repo", str(repo).replace("\\", "/"), "--patch-base", "base",
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "overlaying 2 changed file(s)" in proc.stdout
    patch = out / "patch" / "vllm" / "v1"
    assert sorted(p.name for p in patch.iterdir()) == ["added.py", "changed.py"]
    assert (patch / "changed.py").read_text() == "new\n"
    compose = (out / "mtp-s1-u093" / "compose.yaml").read_text()
    assert "site-packages/vllm/v1/changed.py" in compose
    assert "site-packages/vllm/v1/added.py" in compose
    assert "same.py" not in compose


def test_utils_with_stray_spaces_and_a_fractional_interval(rig):
    proc, out = run_script(
        rig, "--switch", "1", "--no-stress", "--sample-interval", "0.5", utils="  0.93   "
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out / "result.mtp-s1-u093.json").exists()


def test_patch_repo_and_patch_dir_are_exclusive(rig):
    proc, _ = run_script(rig, "--patch-repo", "/x", "--patch-dir", "/y")
    assert proc.returncode == 2


def test_a_peak_above_the_limit_fails_the_run(rig):
    Engine.peak = 32500
    proc, out = run_script(rig, "--switch", "1")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    result = json.loads((out / "result.mtp-s1-u093.json").read_text())
    assert not result["pass"]
    assert any("32500" in r for r in result["fail_reasons"])
    assert "FAIL" in (out / "summary.txt").read_text()


def test_busy_gpus_are_refused(rig):
    (rig["state"] / "used").write_text("20000")
    proc, out = run_script(rig, "--switch", "1")
    assert proc.returncode != 0
    assert "refusing to start" in proc.stderr


@pytest.mark.parametrize(
    "args",
    [[], ["--image", "x"], ["--image", "x", "--models-dir", "/nonexistent-dir"],
     ["--image", "x", "--models-dir", "/tmp", "--switch", "7"],
     ["--image", "x", "--models-dir", "/tmp", "--lane", "weird"]],
)
def test_bad_usage_exits_two(rig, args):
    proc = subprocess.run([BASH, SCRIPT, *args], env=rig["env"], capture_output=True, text=True)
    assert proc.returncode == 2, proc.stdout + proc.stderr


def test_help_mentions_every_option():
    proc = subprocess.run([BASH, SCRIPT, "--help"], capture_output=True, text=True)
    assert proc.returncode == 0
    for option in ("--image", "--models-dir", "--lane", "--utils", "--switch", "--gpus",
                   "--peak-limit-mib", "--headroom-mib", "--no-stress", "--force"):
        assert option in proc.stdout
