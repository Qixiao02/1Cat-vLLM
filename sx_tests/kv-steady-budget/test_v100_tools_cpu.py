# SPDX-License-Identifier: Apache-2.0
"""The tools behind ``run_on_v100.sh``: compose generation, log/sample analysis,
and the stress client. Standard library only; no docker, no GPU.

Run:

    python -m pytest -q sx_tests/kv-steady-budget/test_v100_tools_cpu.py

Asserted:
* compose_gen: the MTP lane has exactly the serve flags of the MTP arm
  (TP4, half, FLASH_ATTN_V100, 32768 / 16 / 8192, util as given, fp16 KV,
  prefix + chunked prefill, tool and reasoning parsers, language-model-only,
  speculative config k=4) and its environment; the no-MTP lane drops the
  speculative config and takes the production shape; the switch, the GPUs, the
  ports and extra environment come from the arguments; bad arguments raise;
  the file is valid YAML when PyYAML is present;
* analyze: peaks per phase and GPU from the samples, KV tokens and available KV
  memory from the engine log (with thousands separators), the engine's own
  steady-budget lines, errors, and a verdict that fails on a peak above the
  limit, on too little CUDA-usable memory at the peak, on failed requests,
  on an OOM line and on an audit that says SHORT;
* stress: unique calibrated prompts, one phase of concurrent requests and one
  single request against a small fake server, the events file, and an exit
  status of 1 when a request fails.
"""

from __future__ import annotations

import http.server
import json
import os
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyze  # noqa: E402
import compose_gen  # noqa: E402
import stress  # noqa: E402

MiB = 1 << 20

# ------------------------------------------------------------------ compose


def render(**over):
    args = dict(
        image="img:tag", tag="mtp-s1-u093", lane="MTP", util="0.93", switch="1",
        gpus="4,5,6,7", models_dir="/models-host", cache_dir="/cache-host", port=8141,
    )
    args.update(over)
    return compose_gen.render(**args)


MTP_FLAGS = [
    "--tensor-parallel-size=4", "--dtype=half", "--attention-backend=FLASH_ATTN_V100",
    "--max-model-len=32768", "--max-num-seqs=16", "--max-num-batched-tokens=8192",
    "--gpu-memory-utilization=0.93", "--kv-cache-dtype=auto", "--trust-remote-code",
    "--enable-prefix-caching", "--enable-chunked-prefill", "--enable-auto-tool-choice",
    "--tool-call-parser=qwen3_coder", "--reasoning-parser=qwen3", "--language-model-only",
    '--speculative-config={"method":"mtp","num_speculative_tokens":4}',
]


def test_mtp_flags_are_the_mtp_arm_in_order():
    flags = compose_gen.build_flags("MTP", "0.93")
    assert flags[0].startswith("/models/")
    served = [f for f in flags if f.startswith("--served-model-name")]
    assert len(served) == 1
    body = [f for f in flags[1:] if not f.startswith(("--served-model-name", "--host", "--port"))]
    assert body == MTP_FLAGS


def test_mtp_environment():
    env = compose_gen.build_env("MTP", "1")
    assert env["VLLM_QWEN4EXP_PLE_HOST_GIB"] == "12"
    assert env["OMP_NUM_THREADS"] == "8"
    assert env["VLLM_SM70_QWEN38_HYBRID_PLE"] == "0"
    assert env["VLLM_PLE_CPU_OFFLOAD"] == "0"
    assert env["VLLM_PLE_DISK_OFFLOAD"] == "0"
    assert env["VLLM_SM70_NVFP4_MOE_GROUPED_DECODE"] == "1"
    assert env["VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS"] == "1200"
    assert env["SX_OPT_KV_STEADY_BUDGET"] == "1"
    assert compose_gen.build_env("MTP", "0")["SX_OPT_KV_STEADY_BUDGET"] == "0"


def test_nomtp_lane_is_the_production_shape_without_speculation():
    flags = compose_gen.build_flags("no-MTP", "0.90")
    assert not any(f.startswith("--speculative-config") for f in flags)
    assert "--max-num-seqs=24" in flags and "--max-model-len=131072" in flags
    assert compose_gen.build_env("no-MTP", "1")["VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS"] == "240"


def test_overrides_and_extras():
    flags = compose_gen.build_flags(
        "MTP", "0.95", max_num_seqs=8, max_model_len=4096, extra_flags=["--foo=1"]
    )
    assert "--max-num-seqs=8" in flags and "--max-model-len=4096" in flags
    assert flags[-1] == "--foo=1"
    env = compose_gen.build_env("MTP", "1", {"OMP_NUM_THREADS": "4", "X": "y"})
    assert env["OMP_NUM_THREADS"] == "4" and env["X"] == "y"


def test_rendered_file_carries_the_arguments():
    text = render()
    assert '"sx-kvsteady-mtp-s1-u093"' in text
    assert '"127.0.0.1:8141:8001"' in text
    assert 'device_ids: ["4", "5", "6", "7"]' in text
    assert 'SX_OPT_KV_STEADY_BUDGET: "1"' in text
    assert '"--gpu-memory-utilization=0.93"' in text
    assert "source: \"/models-host\"" in text and "source: \"/cache-host\"" in text
    assert 'entrypoint: ["vllm", "serve"]' in text
    assert text.endswith("\n") and "\r" not in text


def test_rendered_file_is_valid_yaml_with_the_same_flags():
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(render())
    service = doc["services"]["engine"]
    assert service["command"][1:] == compose_gen.build_flags("MTP", "0.93")[1:]
    assert service["environment"]["SX_OPT_KV_STEADY_BUDGET"] == "1"
    assert service["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] == [
        "4", "5", "6", "7"
    ]
    assert service["ulimits"]["memlock"] == -1


def test_auto_switch_leaves_the_variable_unset():
    assert "SX_OPT_KV_STEADY_BUDGET" not in compose_gen.build_env("MTP", "auto")
    assert "SX_OPT_KV_STEADY_BUDGET" not in compose_gen.build_env("no-MTP", "auto")
    text = render(switch="auto")
    assert "SX_OPT_KV_STEADY_BUDGET:" not in text and "(unset" in text
    assert compose_gen.build_env("MTP", "auto", {"SX_OPT_KV_STEADY_BUDGET": "0"})[
        "SX_OPT_KV_STEADY_BUDGET"
    ] == "0"


def _verdict_for(switch, *, auto_enabled, planned):
    result = {
        "ready": True,
        "switch": switch,
        "memory": {"peak_mib": 31400, "idle_mib": 31000},
        "log": {
            "plan": {"total": str(32510 * 1024 * 1024)} if planned else {},
            "end_of_warmup_free_mib": [],
            "error_lines": [],
            "audit_short_mib": [],
            "auto_enabled": auto_enabled,
        },
    }
    return analyze.verdict(result, peak_limit_mib=32200, headroom_mib=500)


def test_auto_switch_needs_the_auto_enable_line_and_a_plan():
    ok, reasons = _verdict_for("auto", auto_enabled=True, planned=True)
    assert ok, reasons
    ok, reasons = _verdict_for("auto", auto_enabled=False, planned=True)
    assert not ok and any("auto-enable" in r for r in reasons)
    ok, reasons = _verdict_for("auto", auto_enabled=True, planned=False)
    assert not ok and any("never planned" in r for r in reasons)
    ok, reasons = _verdict_for("1", auto_enabled=False, planned=True)
    assert ok, reasons


@pytest.mark.parametrize("over", [{"switch": "2"}, {"gpus": "4,5"}])
def test_bad_arguments_raise(over):
    with pytest.raises(ValueError):
        render(**over)


def test_an_unknown_lane_is_refused():
    with pytest.raises(ValueError, match="lane must be"):
        compose_gen.normalise_lane("weird")
    assert compose_gen.main(["--lane-info", "weird"]) == 2


@pytest.mark.parametrize("raw,expected", [("mtp", "MTP"), ("MTP", "MTP"), ("no-MTP", "no-MTP"),
                                          ("nomtp", "no-MTP"), ("no_mtp", "no-MTP")])
def test_lane_names(raw, expected):
    assert compose_gen.normalise_lane(raw) == expected


def test_cli_writes_the_file(tmp_path):
    out = tmp_path / "c.yaml"
    rc = compose_gen.main([
        "--image", "i", "--tag", "t", "--util", "0.9", "--models-dir", "/m",
        "--cache-dir", "/c", "--env", "A=b", "--serve-arg=--x=1", "--out", str(out),
    ])
    text = out.read_text()
    assert rc == 0 and 'A: "b"' in text and '"--x=1"' in text


def test_overlay_mounts_every_file_under_vllm(tmp_path):
    (tmp_path / "vllm" / "v1" / "worker").mkdir(parents=True)
    (tmp_path / "vllm" / "v1" / "worker" / "kv_steady_budget.py").write_text("x")
    (tmp_path / "vllm" / "config.py").write_text("y")
    (tmp_path / "vllm" / "__pycache__").mkdir()
    (tmp_path / "vllm" / "__pycache__" / "config.cpython-312.pyc").write_text("z")
    (tmp_path / "other.py").write_text("not under vllm")
    files = compose_gen.overlay_files(str(tmp_path))
    assert list(files) == ["vllm/config.py", "vllm/v1/worker/kv_steady_budget.py"]
    text = render(overlay=files, site_packages="/sp")
    for rel, host in files.items():
        assert f'target: "/sp/{rel}", read_only: true' in text
        assert json.dumps(host) in text
    assert "pyc" not in text and "other.py" not in text


def test_overlay_without_a_vllm_directory_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no vllm/"):
        compose_gen.overlay_files(str(tmp_path))


def test_overlay_default_target_is_the_fork_image_layout(tmp_path):
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm" / "a.py").write_text("x")
    text = render(overlay=compose_gen.overlay_files(str(tmp_path)))
    assert "/opt/venv/lib/python3.12/site-packages/vllm/a.py" in text


# ------------------------------------------------------------------ analyze


def write_inputs(tmp_path, *, peak_stress=31811, idle=31499, start_peak=30000, total=32768,
                 log=None, stress_failed=0):
    samples = tmp_path / "samples.csv"
    rows = []
    t = 1000.0
    for step, used in enumerate([2000, start_peak, start_peak, idle, idle, idle,
                                 peak_stress, peak_stress - 20, idle, idle]):
        for gpu in range(4):
            rows.append(f"{t + 2 * step:.2f}, {4 + gpu}, {used - gpu * 20}, {total}")
    samples.write_text("\n".join(rows) + "\n")
    events = tmp_path / "events.log"
    events.write_text(
        "1000.0 compose_up\n1005.5 ready\n1011.5 stress_begin\n1016.5 stress_end\n"
    )
    engine = tmp_path / "engine.log"
    engine.write_text("\n".join(log if log is not None else GOOD_LOG) + "\n")
    stress_json = tmp_path / "stress.json"
    stress_json.write_text(json.dumps({"failed": stress_failed, "total": 7, "requests": []}))
    return samples, events, engine, stress_json


GOOD_LOG = [
    "INFO 10-02 12:00:00 [kv_cache_utils.py:1] GPU KV cache size: 97,000 tokens",
    "INFO 10-02 12:00:00 [kv_cache_utils.py:2] Maximum concurrency for 32,768 tokens per request: 2.96x",
    "(Worker_TP0 pid=11) INFO 10-02 11:59:00 [gpu_worker.py:1] Available KV cache memory: 2.19 GiB",
    "(Worker_TP1 pid=12) INFO 10-02 11:59:00 [gpu_worker.py:1] Available KV cache memory: 2.17 GiB",
    "(Worker_TP0 pid=11) INFO Graph capturing finished in 126 secs, took 1.04 GiB",
    "(Worker_TP0 pid=11) INFO KV steady budget: util 0.9300 of 31.75 GiB",
    "(Worker_TP0 pid=11) INFO KV steady budget [kv=2295000000 total=34089730048 requested=1 "
    "free_after_profile=2 activation=3 graph_reserve=4 post_sizing=5 headroom=6 limiting=physical]",
    "(Worker_TP0 pid=11) INFO KV steady phase end_of_warmup: 1000 MiB free (-5 MiB since the previous phase)",
    "(Worker_TP1 pid=12) INFO KV steady phase end_of_warmup: 1030 MiB free (-5 MiB since the previous phase)",
    "(Worker_TP0 pid=11) INFO KV steady audit: measured post-sizing growth 2500 MiB (plan assumed 2600 MiB); x",
    "(Worker_TP0 pid=11) INFO KV steady audit: OK, 100 MiB above the headroom. Setting "
    "SX_OPT_KV_STEADY_RESERVE_MIB=2900 would size the KV cache exactly to the measurement.",
]


def run_analysis(tmp_path, **kw):
    samples, events, engine, stress_json = write_inputs(tmp_path, **{
        k: v for k, v in kw.items() if k in ("peak_stress", "idle", "start_peak", "total", "log", "stress_failed")})
    out = tmp_path / "result.json"
    rc = analyze.main([
        "run", "--samples", str(samples), "--events", str(events), "--log", str(engine),
        "--stress", str(stress_json), "--tag", "t", "--util", "0.93", "--switch", "1",
        "--out", str(out), *kw.get("extra", []),
    ])
    return rc, json.loads(out.read_text())


def test_phases_and_log_are_parsed(tmp_path):
    rc, result = run_analysis(tmp_path)
    mem = result["memory"]
    assert mem["start_peak_mib"] == 30000
    assert mem["idle_mib"] == 31499
    assert mem["stress_peak_mib"] == 31811
    assert mem["peak_mib"] == 31811
    assert result["memory"]["per_gpu"]["4"]["stress"] == 31811
    assert result["memory"]["per_gpu"]["7"]["stress"] == 31811 - 60
    log = result["log"]
    assert log["kv_tokens"] == 97000
    assert log["available_kv_gib"] == [2.19, 2.17]
    assert log["max_concurrency"] == 2.96
    assert log["graph_capture"] == [{"secs": 126, "gib": 1.04}]
    assert log["plan"]["kv"] == "2295000000" and log["plan"]["total"] == "34089730048"
    assert log["audit_measured_mib"] == [2500.0] and log["audit_planned_mib"] == [2600.0]
    assert log["audit_suggested_reserve_mib"] == [2900] and log["audit_ok_ranks"] == 1
    assert log["error_lines"] == []
    assert rc == 0 and result["pass"] is True


def test_nvidia_smi_is_compared_with_cuda_used_memory(tmp_path):
    rc, result = run_analysis(tmp_path)
    # CUDA total 32510 MiB, least free rank 1000 MiB -> 31510 used; nvidia-smi idle 31499
    assert result["memory"]["smi_minus_cuda_idle_mib"] == 31499 - 31510
    assert result["log"]["end_of_warmup_free_mib"] == [1000, 1030]


def test_only_error_level_lines_and_crash_signatures_are_errors(tmp_path):
    log = GOOD_LOG + [
        "(Worker_TP0 pid=11) INFO a RuntimeError is mentioned in this info line",
        "(Worker_TP0 pid=11) WARNING no error here, just the word error",
        "(Worker_TP0 pid=11) ERROR 10-02 12:00:00 [x.py:1] something failed",
    ]
    rc, result = run_analysis(tmp_path, log=log)
    assert len(result["log"]["error_lines"]) == 1
    assert "something failed" in result["log"]["error_lines"][0]


def test_cuda_usable_total_comes_from_the_engine_log(tmp_path):
    rc, result = run_analysis(tmp_path)
    # 34089730048 B = 32510 MiB; peak 31811 MiB leaves 699 MiB.
    assert result["memory"]["cuda_free_at_peak_mib"] == 699


def test_a_peak_above_the_limit_fails(tmp_path):
    rc, result = run_analysis(tmp_path, peak_stress=32300)
    assert rc == 1 and not result["pass"]
    assert any("peak 32300 MiB > limit 32200 MiB" in r for r in result["fail_reasons"])


def test_the_limit_is_a_parameter(tmp_path):
    rc, result = run_analysis(tmp_path, extra=["--peak-limit-mib", "31000"])
    assert rc == 1
    rc, result = run_analysis(tmp_path, extra=["--peak-limit-mib", "32800", "--headroom-mib", "100"])
    assert rc == 0


def test_too_little_cuda_usable_memory_fails_even_below_the_limit(tmp_path):
    # 32100 MiB < 32200 but only 410 MiB of the 32510 MiB CUDA can hand out is free.
    rc, result = run_analysis(tmp_path, peak_stress=32100)
    assert rc == 1
    assert any("410 MiB of CUDA-usable memory" in r for r in result["fail_reasons"])


def test_oom_in_the_log_fails(tmp_path):
    log = GOOD_LOG + ["(Worker_TP2 pid=13) torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"]
    rc, result = run_analysis(tmp_path, log=log)
    assert rc == 1 and len(result["log"]["error_lines"]) == 1


def test_a_short_audit_fails_and_is_reported(tmp_path):
    log = GOOD_LOG[:-1] + [
        "(Worker_TP0 pid=11) ERROR KV steady audit: SHORT by 300 MiB; the steady peak will leave 200 "
        "MiB free. Set SX_OPT_KV_STEADY_RESERVE_MIB=3100 (or lower --gpu-memory-utilization).",
    ]
    rc, result = run_analysis(tmp_path, log=log)
    assert rc == 1
    assert result["log"]["audit_short_mib"] == [300.0]
    assert result["log"]["audit_suggested_reserve_mib"][-1] == 3100
    assert result["log"]["error_lines"] == []  # the audit has its own reason


def test_failed_requests_fail(tmp_path):
    rc, result = run_analysis(tmp_path, stress_failed=2)
    assert rc == 1 and any("2 of 7 stress requests failed" in r for r in result["fail_reasons"])


def test_engine_that_never_became_healthy_fails(tmp_path):
    samples, events, engine, stress_json = write_inputs(tmp_path)
    events.write_text("1000.0 compose_up\n")
    out = tmp_path / "r.json"
    rc = analyze.main(["run", "--samples", str(samples), "--events", str(events), "--log", str(engine),
                       "--tag", "t", "--util", "0.9", "--switch", "0", "--out", str(out)])
    result = json.loads(out.read_text())
    assert rc == 1 and result["ready"] is False
    assert "engine never became healthy" in result["fail_reasons"]


def test_table(tmp_path, capsys):
    paths = []
    for i, peak in enumerate((31811, 32300)):
        d = tmp_path / str(i)
        d.mkdir()
        samples, events, engine, stress_json = write_inputs(d, peak_stress=peak)
        out = d / "r.json"
        analyze.main(["run", "--samples", str(samples), "--events", str(events), "--log", str(engine),
                      "--stress", str(stress_json), "--tag", f"t{i}", "--util", "0.93", "--switch", "1",
                      "--out", str(out)])
        paths.append(str(out))
    capsys.readouterr()
    rc = analyze.main(["table", *paths])
    text = capsys.readouterr().out
    assert rc == 1
    assert "PASS" in text and "FAIL" in text
    assert "97000" in text and "2.17" in text and "ok (2900)" in text


# ------------------------------------------------------------------- stress


class FakeEngine(http.server.BaseHTTPRequestHandler):
    fail_long = False
    seen: list[int] = []

    def log_message(self, *args):
        pass

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            payload = {"count": len(body["prompt"].split()) + 1}
        else:
            tokens = len(body["prompt"].split()) + 1
            type(self).seen.append(tokens)
            if type(self).fail_long and tokens > 20000:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"CUDA out of memory")
                return
            payload = {"usage": {"prompt_tokens": tokens, "completion_tokens": body["max_tokens"]}}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def engine_port():
    FakeEngine.fail_long = False
    FakeEngine.seen = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeEngine)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()


def test_prompts_are_unique_and_calibrated(engine_port):
    base = f"http://127.0.0.1:{engine_port}"
    a, ta = stress.calibrated_prompt(base, "m", 800, seed=1)
    b, tb = stress.calibrated_prompt(base, "m", 800, seed=2)
    assert a != b
    assert abs(ta - 800) <= 8 and abs(tb - 800) <= 8


def test_stress_run_writes_results_and_events(engine_port, tmp_path):
    out, events = tmp_path / "s.json", tmp_path / "ev.log"
    rc = stress.main([
        "--port", str(engine_port), "--out", str(out), "--events", str(events),
        "--concurrent", "400,400,800", "--single", "1600",
    ])
    result = json.loads(out.read_text())
    assert rc == 0 and result["failed"] == 0 and result["total"] == 4
    assert [r["phase"] for r in result["requests"]] == ["A", "A", "A", "B"]
    assert all(r["status"] == 200 for r in result["requests"])
    names = [line.split()[1] for line in events.read_text().splitlines()]
    assert names == ["stress_begin", "stress_end"]
    assert sorted(FakeEngine.seen)[-1] >= 1500


def test_stress_exit_status_is_one_when_a_request_fails(engine_port, tmp_path):
    FakeEngine.fail_long = True
    out = tmp_path / "s.json"
    rc = stress.main([
        "--port", str(engine_port), "--out", str(out),
        "--concurrent", "400", "--single", "25000",
    ])
    result = json.loads(out.read_text())
    assert rc == 1 and result["failed"] == 1
    assert "out of memory" in result["requests"][-1]["error"]
