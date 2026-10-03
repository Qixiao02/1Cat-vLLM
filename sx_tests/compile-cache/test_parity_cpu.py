# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the cache_parity harness (no docker, no GPU).

    python -m pytest sx_tests/compile-cache/test_parity_cpu.py -q

The orchestration is run end to end against a fake engine: a local HTTP server
that answers /health, /tokenize, /v1/chat/completions and /metrics and a
launcher that writes scripted engine logs. That exercises prompt calibration,
the arm sequence with its cache directories, log parsing, the parity and reuse
gates and the exit codes, for the pass case and for each failure the harness
promises to catch.
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import os
import socket
import sys
import threading

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import cache_parity as cp  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "compose.sample.yaml")


# --------------------------------------------------------------------------
# compose generation
# --------------------------------------------------------------------------


def _compose(lane="nomtp", **over):
    base = open(FIXTURE, encoding="utf-8").read()
    spec = cp.LANES[lane]
    env = {**cp.COMMON_ENV, **spec["env"], "SX_OPT_COMPILE_CACHE": "subgraph"}
    kw = dict(
        name="sx-cp-t-cold",
        container="sx-cp-t-cold",
        image="shixiang/1cat-vllm-v100:test",
        port=8141,
        cache_dir="/mnt/2t/build/cache_parity/caches/cache-on",
        gpus=["4", "5", "6", "7"],
        env=env,
        flags=cp.serve_flags(spec, 8141),
    )
    kw.update(over)
    return cp.compose_from_base(base, **kw)


def test_compose_arm_matches_the_arm_compose_shape():
    import yaml

    doc = yaml.safe_load(_compose())
    assert doc["name"] == "sx-cp-t-cold"
    (svc,) = doc["services"].values()
    assert svc["image"] == "shixiang/1cat-vllm-v100:test"
    assert svc["container_name"] == "sx-cp-t-cold"
    assert svc["restart"] == "no"
    assert svc["ports"] == ["127.0.0.1:8141:8001"]
    assert svc["entrypoint"] == ["vllm", "serve"]
    cmd = svc["command"]
    assert cmd[0] == cp.MODEL_PATH
    for flag in (
        "--tensor-parallel-size=4",
        "--dtype=half",
        "--attention-backend=FLASH_ATTN_V100",
        "--max-model-len=131072",
        "--max-num-seqs=24",
        "--max-num-batched-tokens=8192",
        "--gpu-memory-utilization=0.90",
        "--kv-cache-dtype=auto",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--language-model-only",
    ):
        assert flag in cmd
    assert not any(f.startswith("--speculative-config") for f in cmd)
    env = svc["environment"]
    assert env["SX_OPT_COMPILE_CACHE"] == "subgraph"
    assert env["VLLM_QWEN4EXP_PLE_HOST_GIB"] == "12"
    assert env["VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS"] == "240"
    assert env["OMP_NUM_THREADS"] == "8"
    assert env["XDG_CACHE_HOME"] == "/cache"
    assert "MODEL_PATH" not in env  # the entrypoint-script variables are gone
    volumes = {v["target"]: v for v in svc["volumes"]}
    assert volumes["/cache"]["source"].endswith("cache-on")
    assert "/app/entrypoint.sh" not in volumes and "/usr/local/cuda" not in volumes
    assert volumes["/models"]["read_only"] is True
    devices = svc["deploy"]["resources"]["reservations"]["devices"][0]
    assert devices["device_ids"] == ["4", "5", "6", "7"]
    assert svc["healthcheck"]["test"][0] == "CMD-SHELL"


def test_compose_mtp_lane_flags():
    import yaml

    doc = yaml.safe_load(
        _compose(
            "mtp",
            flags=cp.serve_flags(cp.LANES["mtp"], 8141),
            env={**cp.COMMON_ENV, **cp.LANES["mtp"]["env"]},
        )
    )
    (svc,) = doc["services"].values()
    cmd = svc["command"]
    assert "--max-model-len=32768" in cmd and "--max-num-seqs=16" in cmd
    assert "--gpu-memory-utilization=0.87" in cmd
    assert "--speculative-config=" + cp.MTP_CONFIG in cmd
    assert svc["environment"]["VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS"] == "1200"


def test_serve_flags_drop_and_extra():
    flags = cp.serve_flags(cp.LANES["nomtp"], 8141, drop=["--enable-prefix-caching"], extra=["--foo=1"])
    assert "--enable-prefix-caching" not in flags and flags[-1] == "--foo=1"
    assert "--enable-chunked-prefill" in flags


def test_compose_rejects_an_unknown_template():
    with pytest.raises(ValueError, match="does not match"):
        cp.compose_from_base(
            "name: x\nservices:\n  a:\n    image: y\n",
            name="n",
            container="c",
            image=None,
            port=1,
            cache_dir="/c",
            gpus=["0"],
            env={},
            flags=[],
        )


def test_arm_env_is_deterministic_and_switch_is_last():
    lane = cp.LANES["nomtp"]
    arms = cp.arm_table("aot")
    env = cp.arm_env(lane, arms["warm"], {"VLLM_FOO": "1"})
    assert env["SX_OPT_COMPILE_CACHE"] == "aot" and env["VLLM_FORCE_AOT_LOAD"] == "1"
    assert env["VLLM_FOO"] == "1"
    off = cp.arm_env(lane, arms["off"], {})
    assert off["SX_OPT_COMPILE_CACHE"] == "0" and "VLLM_FORCE_AOT_LOAD" not in off
    assert cp.arm_env(lane, arms["off_noaot"], {})["VLLM_USE_AOT_COMPILE"] == "0"
    # a user env cannot switch the arm's SX_OPT_COMPILE_CACHE
    assert cp.arm_env(lane, arms["cold"], {"SX_OPT_COMPILE_CACHE": "0"})["SX_OPT_COMPILE_CACHE"] == "aot"
    assert cp.arm_table("subgraph")["warm"].extra_env == ()


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------


def _fake_count(text):
    return len(text) // 4  # close to the 4 chars/token start value


def test_prompt_plan_has_ten_prompts_with_unique_ids_and_long_ones():
    plan = cp.prompt_plan((8000, 16000, 32000))
    assert 8 <= len(plan) <= 10 and len({p.id for p in plan}) == len(plan)
    assert sum(1 for p in plan if p.target_tokens >= 8000) >= 4
    assert max(p.target_tokens for p in plan) >= 32000
    assert all(256 <= p.max_tokens <= 512 for p in plan)


def test_calibration_is_deterministic_and_prefixes_do_not_collide():
    plan = cp.prompt_plan((8000, 16000, 32000))
    a = cp.calibrate_prompts(plan, _fake_count)
    b = cp.calibrate_prompts(plan, _fake_count)
    assert [p.user for p in a] == [p.user for p in b]
    for p in a:
        if p.target_tokens:
            assert abs(_fake_count(p.user) - p.target_tokens) <= max(16, p.target_tokens // 100)
    prefixes = {p.user[:40] for p in a}
    assert len(prefixes) == len(a), "a shared prefix would hit the prefix cache"
    # without a tokenizer the 4 chars/token estimate is used
    plain = cp.calibrate_prompts(plan, None)
    assert all(len(p.user) > 0 for p in plain)


def test_needle_document_contains_all_codes_in_order():
    text, codes = cp.needle_document(5, 20000)
    assert len(codes) == 8
    positions = [text.index(code) for code in codes]
    assert positions == sorted(positions)
    assert "Note 1: the vault code number 1 is %s." % codes[0] in text


def test_prompts_roundtrip(tmp_path):
    prompts = cp.calibrate_prompts(cp.prompt_plan((8000, 16000, 28000)), _fake_count)
    path = str(tmp_path / "p.json")
    cp.save_prompts(path, prompts)
    assert cp.load_prompts(path) == prompts
    assert prompts[0].messages()[-1]["role"] == "user"


# --------------------------------------------------------------------------
# log and metrics parsing
# --------------------------------------------------------------------------

COLD_LOG = """\
(EngineCore pid=1) INFO 10-02 19:00:00 [core.py:1] vLLM's torch.compile cache is disabled.
(Worker_TP0 pid=11) INFO 10-02 19:00:01 [backends.py:1] Using cache directory: /cache/vllm/torch_compile_cache/ab/rank_0_0/backbone for vLLM's torch.compile
(Worker_TP0 pid=11) INFO 10-02 19:00:02 [x.py:1] sx-compile-cache build identity 0123456789abcdef (torch 2.10.0+cu128, cuda 12.8, triton 3.6.0, device Tesla V100-SXM2-32GB, torch backport: none)
(Worker_TP0 pid=11) INFO 10-02 19:01:00 [default_loader.py:1] Model loading took 12.3 GiB and 85.123456 seconds
(Worker_TP0 pid=11) INFO 10-02 19:02:00 [backends.py:1] Dynamo bytecode transform time: 40.12 s
(Worker_TP0 pid=11) INFO 10-02 19:03:00 [backends.py:1] Cache the graph of compile range (1, 8192) for later use
(Worker_TP0 pid=11) INFO 10-02 19:03:01 [backends.py:1] Compiling a graph for compile range (1, 8192) takes 51.50 s
(Worker_TP0 pid=11) INFO 10-02 19:03:02 [monitor.py:1] torch.compile and initial profiling/warmup run together took 98.20 s in total
(Worker_TP1 pid=12) INFO 10-02 19:03:02 [monitor.py:1] torch.compile and initial profiling/warmup run together took 97.10 s in total
(Worker_TP0 pid=11) INFO 10-02 19:04:00 [model_runner.py:1] Graph capturing finished in 130 secs, took 1.02 GiB
(Worker_TP0 pid=11) INFO 10-02 19:04:01 [gpu_worker.py:1] SX compile-cache counters: num_models_seen=2 num_graphs_seen=2 num_backend_compilations=6 num_cache_entries_updated=6 num_compiled_artifacts_saved=6 num_compiled_artifacts_loaded=0 num_aot_compiles=0 num_aot_artifacts_saved=0 num_aot_artifacts_loaded=0
(Worker_TP1 pid=12) INFO 10-02 19:04:01 [gpu_worker.py:1] SX compile-cache counters: num_models_seen=2 num_graphs_seen=2 num_backend_compilations=6 num_cache_entries_updated=6 num_compiled_artifacts_saved=6 num_compiled_artifacts_loaded=0 num_aot_compiles=0 num_aot_artifacts_saved=0 num_aot_artifacts_loaded=0
(EngineCore pid=1) INFO 10-02 19:04:02 [core.py:1] init engine (profile, create kv cache, warmup model) took 300.00 s (compilation: 98.20 s)
(EngineCore pid=1) INFO 10-02 19:04:02 [kv_cache_utils.py:1] GPU KV cache size: 410,247 tokens
ERROR 10-02 19:04:03 [foo.py:1] something harmless
"""

WARM_LOG = """\
(Worker_TP0 pid=21) INFO 10-02 20:00:00 [compiler_interface.py:1] Directly load the compiled graph(s) for compile range (1, 8192) from the cache, took 2.500 s
(Worker_TP0 pid=21) INFO 10-02 20:00:01 [decorators.py:1] Directly load AOT compilation from path /cache/vllm/torch_compile_cache/torch_aot_compile/x/rank_0_0/model
(Worker_TP0 pid=21) INFO 10-02 20:00:02 [monitor.py:1] torch.compile took 9.20 s in total
(Worker_TP0 pid=21) WARNING 10-02 20:00:03 [decorators.py:1] Compiling model again due to a load failure from /cache/x, reason: boom
(Worker_TP0 pid=21) INFO 10-02 20:00:04 [gpu_worker.py:1] SX compile-cache counters: num_models_seen=2 num_graphs_seen=2 num_backend_compilations=6 num_cache_entries_updated=0 num_compiled_artifacts_saved=0 num_compiled_artifacts_loaded=6 num_aot_compiles=0 num_aot_artifacts_saved=0 num_aot_artifacts_loaded=2
"""


def test_parse_engine_log_cold():
    parsed = cp.parse_engine_log(COLD_LOG)
    assert parsed["model_loading_s"] == [85.123456]
    assert parsed["dynamo_s"] == [40.12]
    assert parsed["compile_range_s"] == [51.5]
    assert parsed["torch_compile_warmup_s"] == [98.2, 97.1]
    assert parsed["graph_capture_s"] == [130.0]
    assert parsed["init_engine_s"] == [300.0]
    assert parsed["kv_cache_tokens"] == 410247
    assert parsed["identity"] == "0123456789abcdef"
    assert parsed["cache_disabled_notice"] == 1 and parsed["graphs_cached"] == 1
    assert set(parsed["counters"]) == {"Worker_TP0", "Worker_TP1"}
    assert parsed["bad_lines"] == []
    assert len(parsed["errors"]) == 1 and "harmless" in parsed["errors"][0]
    phases = cp.phase_summary(parsed)
    assert phases["torch_compile_total_s"] == 98.2  # slowest rank, not the sum of ranks
    assert phases["graph_capture_max_s"] == 130.0
    reuse = cp.reuse_stats(parsed)
    assert reuse["num_compiled_artifacts_saved"] == 12  # summed over the two ranks


def test_parse_engine_log_warm_flags_load_failure():
    parsed = cp.parse_engine_log(WARM_LOG)
    assert parsed["cache_load_s"] == [2.5] and parsed["aot_loaded"] == 1
    assert parsed["torch_compile_s"] == [9.2]
    assert any("load failure" in line for line in parsed["bad_lines"])
    assert cp.reuse_stats(parsed)["num_compiled_artifacts_loaded"] == 6


METRICS = """\
# HELP vllm:spec_decode_num_drafts_total Number of spec decoding drafts.
vllm:spec_decode_num_drafts_total{engine="0",model_name="m"} 100.0
vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="m"} 400.0
vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="m"} 250.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="0"} 90.0
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",model_name="m",position="1"} 70.0
vllm:num_requests_running{engine="0"} 0.0
"""


def test_spec_metrics_parse_and_delta():
    before = cp.parse_spec_metrics(METRICS)
    assert before["num_drafts"] == 100 and before["num_accepted_tokens"] == 250
    assert before["num_accepted_tokens_per_pos[1]"] == 70
    after = cp.parse_spec_metrics(METRICS.replace("100.0", "130.0").replace("250.0", "310.0"))
    delta = cp.spec_delta(before, after)
    assert delta["num_drafts"] == 30 and delta["num_accepted_tokens"] == 60
    assert delta["num_draft_tokens"] == 0


# --------------------------------------------------------------------------
# comparison and verdicts
# --------------------------------------------------------------------------


def _arm(tokens_by_prompt, reuse=None, spec=None, failed=None, bad=None):
    outputs = [
        {"id": pid, "ok": True, "token_ids": list(t), "text": "".join(map(chr, [65 + x % 26 for x in t])), "completion_tokens": len(t)}
        for pid, t in tokens_by_prompt.items()
    ]
    arm = {"outputs": outputs, "reuse": reuse or {}, "log": {"bad_lines": bad or [], "errors": []}, "phases": {}, "healthy_s": 500.0}
    if spec is not None:
        arm["spec"] = spec
    if failed:
        arm["failed"] = failed
    return arm


TOK = {"a": range(50), "b": range(100, 160)}
COLD_REUSE = {"num_compiled_artifacts_saved": 6, "num_compiled_artifacts_loaded": 0, "num_aot_compiles": 0}
WARM_REUSE = {"num_compiled_artifacts_saved": 0, "num_compiled_artifacts_loaded": 6, "num_aot_compiles": 0}


def test_identical_arms_pass():
    results = {
        "off": _arm(TOK),
        "cold": _arm(TOK, COLD_REUSE),
        "warm": _arm(TOK, WARM_REUSE),
    }
    report, code = cp.build_report(results, "subgraph", "nomtp")
    assert code == 0 and report["verdict"] == "PASS"
    assert report["reasons"] == []
    assert all(p["identical"] for p in report["pairs"])
    assert "PASS" in cp.format_report(report)


def test_one_token_difference_fails_loudly_with_position():
    drift = {k: list(v) for k, v in TOK.items()}
    drift["b"][37] += 1
    results = {"off": _arm(TOK), "cold": _arm(TOK, COLD_REUSE), "warm": _arm(drift, WARM_REUSE)}
    report, code = cp.build_report(results, "subgraph", "nomtp")
    assert code == 1 and report["verdict"] == "FAIL"
    pair = next(p for p in report["pairs"] if (p["a"], p["b"]) == ("off", "warm"))
    assert pair["equal"] == 1 and pair["total"] == 2
    assert pair["differences"][0]["position"] == 37
    assert "token 37" in pair["differences"][0]["reason"]
    # cold vs warm differs too; off vs cold does not
    assert next(p for p in report["pairs"] if (p["a"], p["b"]) == ("off", "cold"))["identical"]
    text = cp.format_report(report)
    assert "FAILED" in text and "off vs warm" in text


def test_length_difference_is_a_difference():
    short = {k: list(v) for k, v in TOK.items()}
    short["a"] = short["a"][:-1]  # stops one token early
    res = cp.compare_pair("off", _arm(TOK), "warm", _arm(short))
    assert not res.identical and res.differences[0]["position"] == 49


def test_draft_counters_must_match_exactly():
    spec = {"num_drafts": 30.0, "num_draft_tokens": 120.0, "num_accepted_tokens": 70.0, "num_accepted_tokens_per_pos[0]": 25.0}
    other = dict(spec, num_accepted_tokens=69.0)
    results = {
        "off": _arm(TOK, spec=spec),
        "cold": _arm(TOK, COLD_REUSE, spec=spec),
        "warm": _arm(TOK, WARM_REUSE, spec=other),
    }
    report, code = cp.build_report(results, "subgraph", "mtp")
    assert code == 1
    pair = next(p for p in report["pairs"] if (p["a"], p["b"]) == ("off", "warm"))
    assert pair["equal"] == pair["total"] and pair["spec_equal"] is False
    assert pair["spec_difference"]["num_accepted_tokens"] == (70.0, 69.0)
    # a missing spec dict on one side is not a pass or a fail by itself
    assert cp.compare_pair("a", _arm(TOK, spec=spec), "b", _arm(TOK)).spec_equal is None


@pytest.mark.parametrize(
    "warm_reuse,fragment",
    [
        ({"num_compiled_artifacts_saved": 3, "num_compiled_artifacts_loaded": 3}, "compiled and saved 3"),
        ({"num_compiled_artifacts_saved": 0, "num_compiled_artifacts_loaded": 0}, "loaded no compiled artifact"),
    ],
)
def test_warm_without_reuse_fails_even_if_outputs_match(warm_reuse, fragment):
    results = {"off": _arm(TOK), "cold": _arm(TOK, COLD_REUSE), "warm": _arm(TOK, warm_reuse)}
    report, code = cp.build_report(results, "subgraph", "nomtp")
    assert code == 1 and any(fragment in r for r in report["reasons"])


def test_aot_mode_reuse_rules():
    cold = {"num_aot_artifacts_saved": 2, "num_aot_artifacts_loaded": 0, "num_aot_compiles": 2}
    ok = {"num_aot_artifacts_saved": 0, "num_aot_artifacts_loaded": 2, "num_aot_compiles": 0}
    assert cp.reuse_verdict({"reuse": cold}, {"reuse": ok}, "aot") == []
    bad = dict(ok, num_aot_artifacts_loaded=1, num_aot_compiles=1)
    problems = cp.reuse_verdict({"reuse": cold}, {"reuse": bad}, "aot")
    assert len(problems) == 2
    assert cp.reuse_verdict({"reuse": {}}, {"reuse": ok}, "aot")  # no counters at all


def test_load_failure_in_the_warm_log_fails_the_run():
    bad = ["Compiling model again due to a load failure from /cache/x, reason: boom"]
    results = {"off": _arm(TOK), "cold": _arm(TOK, COLD_REUSE), "warm": _arm(TOK, WARM_REUSE, bad=bad)}
    report, code = cp.build_report(results, "aot", "nomtp")
    assert code == 1 and any("load failure" in r for r in report["reasons"])


def test_unreproducible_reference_is_called_out():
    drift = {k: list(v) for k, v in TOK.items()}
    drift["a"][3] += 1
    results = {
        "off": _arm(TOK),
        "off2": _arm(drift),
        "cold": _arm(drift, COLD_REUSE),
        "warm": _arm(drift, WARM_REUSE),
    }
    report, code = cp.build_report(results, "subgraph", "nomtp")
    assert code == 1  # off vs cold differs
    assert any("reference itself" in n for n in report["notes"])
    assert not report["controls"][0]["identical"]


def test_failed_arm_and_failed_request_are_reported():
    results = {"off": _arm(TOK, failed="engine exited before becoming healthy")}
    results["off"]["outputs"] = []
    report, code = cp.build_report(results, "subgraph", "nomtp")
    assert code == 1 and any("arm off failed" in r for r in report["reasons"])
    bad_request = _arm(TOK)
    bad_request["outputs"][0] = {"id": "a", "ok": False, "error": "timeout"}
    report, code = cp.build_report({"off": _arm(TOK), "cold": bad_request}, "subgraph", "nomtp")
    assert code == 1 and any("request a failed" in r for r in report["reasons"])


def test_text_fallback_when_the_api_returns_no_token_ids():
    a = {"outputs": [{"id": "x", "ok": True, "token_ids": None, "text": "hello world"}]}
    b = {"outputs": [{"id": "x", "ok": True, "token_ids": None, "text": "hello w0rld"}]}
    res = cp.compare_pair("a", a, "b", b)
    assert not res.identical and "char 7" in res.differences[0]["reason"]


# --------------------------------------------------------------------------
# end to end against a fake engine
# --------------------------------------------------------------------------


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Behaviour:
    """What the fake engine does per arm; changed by the test."""

    def __init__(self):
        self.drift_arms = {}  # arm -> token index to corrupt
        self.logs = {}
        self.fail_arm = None
        self.requests = []
        self.spec = False


class FakeEngine:
    def __init__(self, behaviour, port):
        self.b = behaviour
        self.port = port
        self.base_url = "http://127.0.0.1:%d" % port
        self.arm = None
        self.server = None
        self.thread = None
        self.started = []
        self.stopped = 0
        self.cache_dirs = []

    def start(self, arm, env, cache_dir):
        self.arm = arm.name
        self.started.append((arm.name, dict(env), cache_dir))
        self.cache_dirs.append(cache_dir)
        if self.b.fail_arm == arm.name:
            return
        behaviour, name = self.b, arm.name
        counters = {"drafts": 0.0, "accepted": 0.0}

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/health":
                    self._send(200, b"ok", "text/plain")
                elif self.path == "/metrics":
                    text = "vllm:spec_decode_num_drafts_total{engine=\"0\"} %s\nvllm:spec_decode_num_accepted_tokens_total{engine=\"0\"} %s\n" % (counters["drafts"], counters["accepted"])
                    self._send(200, text.encode(), "text/plain")
                else:
                    self._send(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/tokenize":
                    self._send(200, {"count": len(body["prompt"]) // 4})
                    return
                text = json.dumps(body["messages"])
                behaviour.requests.append((name, body["max_tokens"]))
                seed = hashlib.sha256(text.encode()).digest()
                n = min(body["max_tokens"], 24)
                ids = [seed[i % 32] * 7 + i for i in range(n)]
                if name in behaviour.drift_arms:
                    ids[behaviour.drift_arms[name] % n] += 1
                counters["drafts"] += n / 4
                counters["accepted"] += n / 2 - (1 if name in behaviour.drift_arms else 0)
                self._send(200, {
                    "choices": [{"finish_reason": "stop", "token_ids": ids,
                                 "message": {"content": " ".join(map(str, ids))}}],
                    "usage": {"completion_tokens": n, "prompt_tokens": len(text) // 4},
                })

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), H)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def alive(self):
        return self.b.fail_arm != self.arm

    def logs(self):
        return self.b.logs.get(self.arm, "")

    def stop(self):
        self.stopped += 1
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None


def _args(tmp_path, port, **over):
    ns = argparse.Namespace(
        lane="nomtp", cache_mode="subgraph", arms="off,cold,warm", compose="", image="img",
        engine_script=None, gpus="4,5,6,7", port=port, tag="t", workdir=str(tmp_path / "w"),
        cache_root=str(tmp_path / "caches"), cache_seed=None, env=[], flag=[], drop_flag=[],
        prompts_json=None, health_timeout=20, request_timeout=20, stop_first=None,
        restart_after=None, dry_run=False,
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


@pytest.fixture
def fake(monkeypatch, tmp_path):
    port = _free_port()
    behaviour = _Behaviour()
    behaviour.logs = {"off": COLD_LOG.replace("num_compiled_artifacts_saved=6", "num_compiled_artifacts_saved=0"),
                      "cold": COLD_LOG, "warm": WARM_LOG.replace("Compiling model again due to a load failure from /cache/x, reason: boom", "ok")}
    monkeypatch.setattr(cp, "wait_gpus_free", lambda *a, **k: None)
    monkeypatch.setattr(cp.time, "sleep", lambda s: None)
    return behaviour, port, tmp_path


def test_end_to_end_pass(fake):
    behaviour, port, tmp_path = fake
    args = _args(tmp_path, port)
    engine = FakeEngine(behaviour, port)
    code = cp.cmd_run(args, engine=engine)
    assert code == 0
    assert [a for a, _, _ in engine.started] == ["off", "cold", "warm"]
    # arm environments and cache directories
    envs = {a: e for a, e, _ in engine.started}
    assert envs["off"]["SX_OPT_COMPILE_CACHE"] == "0"
    assert envs["cold"]["SX_OPT_COMPILE_CACHE"] == envs["warm"]["SX_OPT_COMPILE_CACHE"] == "subgraph"
    dirs = {a: d for a, _, d in engine.started}
    assert dirs["cold"] == dirs["warm"] != dirs["off"]
    assert dirs["off"].endswith("cache-off") and dirs["cold"].endswith("cache-on")
    assert engine.stopped == 3
    # identical prompt set, every prompt sent in every arm, files written
    workdir = args.workdir
    prompts = cp.load_prompts(os.path.join(workdir, "prompts.json"))
    assert len(prompts) == 10
    for name in ("off", "cold", "warm"):
        saved = json.load(open(os.path.join(workdir, "arm_%s.json" % name)))
        assert len(saved["outputs"]) == 10 and saved["healthy_s"] >= 0
        assert os.path.exists(os.path.join(workdir, "engine_%s.log" % name))
    assert open(os.path.join(workdir, "report.md")).read().startswith("cache_parity lane=nomtp mode=subgraph: PASS")
    assert json.load(open(os.path.join(workdir, "report.json")))["verdict"] == "PASS"


def test_end_to_end_cold_cache_dir_is_wiped_and_warm_keeps_it(fake):
    behaviour, port, tmp_path = fake
    args = _args(tmp_path, port, arms="cold,warm")
    cold_dir = os.path.join(args.cache_root, "cache-on")
    os.makedirs(cold_dir)
    open(os.path.join(cold_dir, "stale"), "w").write("x")
    engine = FakeEngine(behaviour, port)

    started_warm = {}
    real_start = engine.start

    def start(arm, env, cache_dir):
        if arm.name == "warm":
            started_warm["files"] = os.listdir(cache_dir)
        else:
            assert not os.path.exists(os.path.join(cache_dir, "stale"))
            open(os.path.join(cache_dir, "written-by-cold"), "w").write("y")
        real_start(arm, env, cache_dir)

    engine.start = start
    cp.cmd_run(args, engine=engine)
    assert started_warm["files"] == ["written-by-cold"]


def test_end_to_end_drift_in_the_warm_arm_fails(fake):
    behaviour, port, tmp_path = fake
    behaviour.drift_arms = {"warm": 13}
    args = _args(tmp_path, port)
    code = cp.cmd_run(args, engine=FakeEngine(behaviour, port))
    assert code == 1
    report = json.load(open(os.path.join(args.workdir, "report.json")))
    assert report["verdict"] == "FAIL"
    assert any("off vs warm" in r for r in report["reasons"])
    assert not any("off vs cold" in r for r in report["reasons"])


def test_end_to_end_without_reuse_fails(fake):
    behaviour, port, tmp_path = fake
    behaviour.logs["warm"] = COLD_LOG  # the "warm" engine compiled and saved again
    args = _args(tmp_path, port)
    assert cp.cmd_run(args, engine=FakeEngine(behaviour, port)) == 1
    report = json.load(open(os.path.join(args.workdir, "report.json")))
    assert any(r.startswith("reuse:") for r in report["reasons"])


def test_end_to_end_engine_that_never_starts_stops_the_chain(fake):
    behaviour, port, tmp_path = fake
    behaviour.fail_arm = "cold"
    args = _args(tmp_path, port)
    engine = FakeEngine(behaviour, port)
    assert cp.cmd_run(args, engine=engine) == 1
    assert [a for a, _, _ in engine.started] == ["off", "cold"]  # warm never ran
    report = json.load(open(os.path.join(args.workdir, "report.json")))
    assert any("arm cold failed" in r for r in report["reasons"])


def test_end_to_end_mtp_lane_records_draft_counters(fake):
    behaviour, port, tmp_path = fake
    args = _args(tmp_path, port, lane="mtp", arms="off,cold,warm")
    engine = FakeEngine(behaviour, port)
    assert cp.cmd_run(args, engine=engine) == 0
    saved = json.load(open(os.path.join(args.workdir, "arm_warm.json")))
    assert saved["spec"]["num_drafts"] > 0
    # the long prompts of the MTP lane fit its 32K context
    prompts = cp.load_prompts(os.path.join(args.workdir, "prompts.json"))
    assert max(p.target_tokens for p in prompts) + 512 <= 32768


def test_end_to_end_control_arms_are_compared(fake):
    behaviour, port, tmp_path = fake
    behaviour.logs["off2"] = behaviour.logs["off"]
    args = _args(tmp_path, port, arms="off,off2,cold,warm")
    engine = FakeEngine(behaviour, port)
    assert cp.cmd_run(args, engine=engine) == 0
    dirs = {a: d for a, _, d in engine.started}
    assert dirs["off"] == dirs["off2"]
    report = json.load(open(os.path.join(args.workdir, "report.json")))
    assert report["controls"] and report["controls"][0]["identical"]


def test_unknown_arm_and_missing_cold_are_rejected(fake, capsys):
    behaviour, port, tmp_path = fake
    assert cp.cmd_run(_args(tmp_path, port, arms="off,bogus"), engine=FakeEngine(behaviour, port)) == 2
    assert cp.cmd_run(_args(tmp_path, port, arms="warm"), engine=FakeEngine(behaviour, port)) == 2


def test_compare_subcommand_reads_saved_arms(tmp_path):
    for name, tokens, reuse in (("off", TOK, None), ("cold", TOK, COLD_REUSE), ("warm", TOK, WARM_REUSE)):
        (tmp_path / ("%s.json" % name)).write_text(json.dumps(_arm(tokens, reuse)))
    argv = ["compare", "--cache-mode", "subgraph"]
    for name in ("off", "cold", "warm"):
        argv += ["--arm", "%s=%s" % (name, tmp_path / ("%s.json" % name))]
    assert cp.main(argv) == 0


def test_prompts_subcommand_writes_a_set(tmp_path, capsys):
    out = tmp_path / "p.json"
    assert cp.main(["prompts", "--out", str(out), "--lane", "mtp"]) == 0
    assert len(cp.load_prompts(str(out))) == 10
    assert "needle" in capsys.readouterr().out


def test_dry_run_only_writes_compose_files(tmp_path, monkeypatch):
    behaviour = _Behaviour()
    args = _args(tmp_path, 1, compose=FIXTURE, dry_run=True, arms="off,cold")
    code = cp.cmd_run(args)
    assert code == 0
    for name in ("off", "cold"):
        text = open(os.path.join(args.workdir, "compose.%s.yaml" % name)).read()
        assert "SX_OPT_COMPILE_CACHE" in text and "vllm" in text
    assert 'SX_OPT_COMPILE_CACHE: "0"' in open(os.path.join(args.workdir, "compose.off.yaml")).read()
    assert 'SX_OPT_COMPILE_CACHE: "subgraph"' in open(os.path.join(args.workdir, "compose.cold.yaml")).read()


def test_invalidation_arm_must_recompile():
    base = {"off": _arm(TOK), "cold": _arm(TOK, COLD_REUSE), "warm": _arm(TOK, WARM_REUSE)}
    recompiled = dict(base, inval=_arm(TOK, COLD_REUSE))
    report, code = cp.build_report(recompiled, "subgraph", "nomtp")
    assert code == 0 and ("cold", "inval") in [(p["a"], p["b"]) for p in report["pairs"]]
    reused = dict(base, inval=_arm(TOK, WARM_REUSE))
    report, code = cp.build_report(reused, "subgraph", "nomtp")
    assert code == 1 and any("invalidation" in r for r in report["reasons"])
    # the probe variable is part of the arm, not of the other arms
    table = cp.arm_table("subgraph")
    assert dict(table["inval"].extra_env) == {"VLLM_SX_CACHE_PROBE": "1"}
    assert "VLLM_SX_CACHE_PROBE" not in cp.arm_env(cp.LANES["nomtp"], table["warm"], {})


def test_wipe_dir_falls_back_when_the_container_owned_the_files(tmp_path, monkeypatch):
    real_rmtree = cp.shutil.rmtree
    path = tmp_path / "cache-on"
    messages = []
    removed = []

    def denied(p, ignore_errors=False, **kw):
        # the operator cannot delete files created by the container user
        if ignore_errors:
            return None
        raise PermissionError("owned by uid 997")

    monkeypatch.setattr(cp.shutil, "rmtree", denied)
    monkeypatch.setattr(cp.shutil, "which", lambda name: None)  # no sudo, no docker
    path.mkdir()
    with pytest.raises(cp.EngineError, match="remove it by hand"):
        cp.wipe_dir(str(path), None, messages.append)
    assert any("rmtree" in m for m in messages)

    # with docker, a root container does the removal
    def root_container(cmd, **kw):
        removed.append(cmd)
        real_rmtree(str(path))
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(cp.shutil, "which", lambda name: "/usr/bin/" + name if name == "docker" else None)
    monkeypatch.setattr(cp, "run", root_container)
    cp.wipe_dir(str(path), "img", messages.append)
    assert not path.exists()
    assert removed and "--user" in removed[0] and "img" in removed[0]
    # the plain case needs no helper at all
    monkeypatch.undo()
    path.mkdir()
    (path / "f").write_text("x")
    cp.wipe_dir(str(path), None, messages.append)
    assert not path.exists()
