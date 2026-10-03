# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the SX_OPT_COMPILE_CACHE switch (no GPU, no vLLM install).

    python -m pytest sx_tests/compile-cache/test_switch_cpu.py -q

Run this directory in its own pytest process: the boot helper registers bare
``vllm`` packages in ``sys.modules`` and must not meet another directory's.

What is checked
  * mode parsing and the policy table (plan_policy / apply_policy);
  * the getters in vllm/envs.py (disable_compile_cache, use_aot_compile,
    compile_factors) cut out of the real source and exec'd against stubs, and,
    with the switch off, compared with the same functions at BASE_REV
    ("0 keeps today's behaviour exactly");
  * the lane block of VllmConfig.__post_init__ (cache opt-out and AOT default),
    old text at BASE_REV versus the current text, for every switch value and
    explicit-env combination;
  * the cache key: every VLLM_/SX_OPT_ switch, the build identity (source
    content, versions, native libraries, torch backport level, kernel-relevant
    env), the baked constants and the checkpoint identity change it.
"""

from __future__ import annotations

import itertools
import os
import sys
import textwrap
import types

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import cache_boot as boot  # noqa: E402

sx = boot.load_sx_module()
boot.silence_sx_logger()


# --------------------------------------------------------------------------
# modes and policy
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,mode",
    [
        (None, "off"),
        ("", "off"),
        ("0", "off"),
        ("off", "off"),
        (" False ", "off"),
        ("1", "subgraph"),
        ("on", "subgraph"),
        ("TRUE", "subgraph"),
        ("subgraph", "subgraph"),
        ("aot", "aot"),
        ("AOT", "aot"),
        ("2", "off"),  # unknown values never enable the reuse
        ("yes please", "off"),
    ],
)
def test_mode_parsing(raw, mode):
    env = {} if raw is None else {"SX_OPT_COMPILE_CACHE": raw}
    assert sx.compile_cache_mode(env) == mode
    assert sx.compile_cache_enabled(env) == (mode != "off")


def test_default_is_off_and_plans_nothing():
    policy = sx.plan_policy({})
    assert policy.mode == "off"
    assert policy.sets == () and policy.messages == ()
    assert policy.aot is None


def test_subgraph_policy_disables_aot_reload_and_keeps_cache_on():
    policy = sx.plan_policy({"SX_OPT_COMPILE_CACHE": "1"})
    assert policy.mode == "subgraph"
    assert not policy.cache_disabled
    assert policy.aot is False
    assert policy.sets == (("VLLM_USE_AOT_COMPILE", "0"),)
    assert any("without" in text and "AOT" in text for _, text in policy.messages)


def test_aot_policy_reloads_the_whole_artifact():
    policy = sx.plan_policy({"SX_OPT_COMPILE_CACHE": "aot"})
    assert policy.mode == "aot" and policy.aot is True
    assert policy.sets == (("VLLM_USE_AOT_COMPILE", "1"),)


def test_explicit_env_always_wins():
    # explicit opt-out beats the switch (and says so)
    policy = sx.plan_policy(
        {"SX_OPT_COMPILE_CACHE": "1", "VLLM_DISABLE_COMPILE_CACHE": "1"}
    )
    assert policy.cache_disabled
    assert any(level == "warning" for level, _ in policy.messages)
    # explicit =0 is the same as unset
    policy = sx.plan_policy(
        {"SX_OPT_COMPILE_CACHE": "1", "VLLM_DISABLE_COMPILE_CACHE": "0"}
    )
    assert not policy.cache_disabled
    # an explicit AOT choice is never overwritten
    for explicit, switch, warns in [
        ("1", "1", True),  # reload requested through the subgraph switch
        ("1", "aot", False),
        ("0", "1", False),
        ("0", "aot", True),  # aot switch with AOT forced off
    ]:
        env = {"SX_OPT_COMPILE_CACHE": switch, "VLLM_USE_AOT_COMPILE": explicit}
        policy = sx.plan_policy(env)
        assert policy.sets == (), (explicit, switch)
        assert policy.aot == (explicit == "1")
        assert any(level == "warning" for level, _ in policy.messages) == warns, (
            explicit,
            switch,
        )


def test_apply_policy_uses_setdefault_and_logs():
    env = {"SX_OPT_COMPILE_CACHE": "1"}
    log = boot.RecordingLogger()
    sx.apply_policy(sx.plan_policy(env), env, log)
    assert env["VLLM_USE_AOT_COMPILE"] == "0"
    assert log.records and all(level in ("info", "warning") for level, _ in log.records)
    # a value that appeared in between (another thread/process) survives
    env2 = {"SX_OPT_COMPILE_CACHE": "1"}
    policy = sx.plan_policy(env2)
    env2["VLLM_USE_AOT_COMPILE"] = "1"
    sx.apply_policy(policy, env2, boot.RecordingLogger())
    assert env2["VLLM_USE_AOT_COMPILE"] == "1"


# --------------------------------------------------------------------------
# envs.py getters
# --------------------------------------------------------------------------


def _envs_namespace(source: str, environ: dict) -> dict:
    """disable_compile_cache / use_aot_compile / compile_factors from source."""
    os_stub = types.SimpleNamespace(
        environ=environ,
        getenv=lambda k, d=None: environ.get(k, d),
    )
    torch_utils = types.ModuleType("vllm.utils.torch_utils")
    torch_utils.is_torch_equal_or_newer = lambda version: True
    config_utils = types.ModuleType("vllm.config.utils")
    config_utils.normalize_value = lambda x: x
    for name in ("vllm.utils", "vllm.config"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["vllm.utils.torch_utils"] = torch_utils
    sys.modules["vllm.config.utils"] = config_utils
    # sx_compile_cache reads os.environ itself: point it at the stub mapping
    namespace = {
        "os": os_stub,
        "logger": boot.RecordingLogger(),
        "environment_variables": {
            "VLLM_REGISTERED": lambda: 1,
            "VLLM_PORT": lambda: 8000,
        },
    }
    boot.exec_functions(
        source,
        ["disable_compile_cache", "use_aot_compile", "compile_factors"],
        namespace,
    )
    return namespace


@pytest.fixture
def real_environ(monkeypatch):
    """os.environ cleared of the switches, restored afterwards."""
    for name in list(os.environ):
        if name.startswith(("VLLM_", "SX_OPT_")):
            monkeypatch.delenv(name)
    return os.environ


def _current_ns(environ):
    return _envs_namespace(boot.read(boot.ENVS), environ)


def _base_ns(environ):
    old = boot.git_show(boot.BASE_REV, "vllm/envs.py")
    if old is None:
        pytest.skip("git history with the base revision is not available")
    return _envs_namespace(old, environ)


LANE = "VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH"
GETTER_CASES = [
    {},
    {LANE: "1"},
    {LANE: "1", "VLLM_DISABLE_COMPILE_CACHE": "0"},
    {LANE: "1", "VLLM_DISABLE_COMPILE_CACHE": "1"},
    {LANE: "1", "VLLM_USE_AOT_COMPILE": "0"},
    {LANE: "1", "VLLM_USE_AOT_COMPILE": "1", "VLLM_DISABLE_COMPILE_CACHE": "0"},
    {"VLLM_DISABLE_COMPILE_CACHE": "1"},
    {"VLLM_USE_AOT_COMPILE": "1"},
]


def _getters(ns):
    return ns["disable_compile_cache"](), ns["use_aot_compile"]()


@pytest.mark.parametrize("off_value", [None, "0", "off", "false", ""])
@pytest.mark.parametrize("case", GETTER_CASES)
def test_getters_off_equal_base(case, off_value, real_environ, monkeypatch):
    for k, v in case.items():
        monkeypatch.setenv(k, v)
    if off_value is not None:
        monkeypatch.setenv("SX_OPT_COMPILE_CACHE", off_value)
    assert _getters(_current_ns(os.environ)) == _getters(_base_ns(os.environ))


def test_getters_on(real_environ, monkeypatch):
    monkeypatch.setenv(LANE, "1")
    ns = _current_ns(os.environ)
    assert _getters(ns) == (True, True)  # today: opt-out forced, in-memory AOT
    monkeypatch.setenv("SX_OPT_COMPILE_CACHE", "1")
    assert _getters(ns) == (False, False)  # cache on, subgraph reuse
    monkeypatch.setenv("SX_OPT_COMPILE_CACHE", "aot")
    assert _getters(ns) == (False, True)
    # explicit variables win over the switch
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "0")
    assert _getters(ns) == (True, False)


def test_compile_factors_off_equals_base(real_environ, monkeypatch):
    monkeypatch.setenv("VLLM_UNREGISTERED_KERNEL_SWITCH", "1")
    monkeypatch.setenv("SX_OPT_MTP_ROWS", "0")
    base = _base_ns(os.environ)["compile_factors"]()
    assert _current_ns(os.environ)["compile_factors"]() == base
    for off in ("0", "off", ""):
        monkeypatch.setenv("SX_OPT_COMPILE_CACHE", off)
        assert _current_ns(os.environ)["compile_factors"]() == base


@pytest.fixture
def fixed_identity(monkeypatch):
    """A deterministic build identity so the tests do not hash the tree."""
    state = {"value": "build-A"}
    monkeypatch.setattr(sx, "build_identity", lambda: state["value"])
    sx.clear_baked_constants()
    yield state
    sx.clear_baked_constants()


def test_compile_factors_on_carry_identity_and_switches(
    real_environ, monkeypatch, fixed_identity
):
    monkeypatch.setenv("SX_OPT_COMPILE_CACHE", "1")
    ns = _current_ns(os.environ)
    factors = ns["compile_factors"]()
    assert factors["SX_OPT_COMPILE_CACHE"] == "subgraph"
    assert factors["SX_COMPILE_CACHE_BUILD"] == "build-A"
    assert factors["SX_COMPILE_CACHE_BAKED"] == {}

    # spelling of the switch does not matter, its meaning does
    monkeypatch.setenv("SX_OPT_COMPILE_CACHE", "on")
    assert ns["compile_factors"]() == factors
    monkeypatch.setenv("SX_OPT_COMPILE_CACHE", "aot")
    assert ns["compile_factors"]() != factors
    monkeypatch.setenv("SX_OPT_COMPILE_CACHE", "1")

    def changed(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        try:
            return ns["compile_factors"]() != factors
        finally:
            for k in env:
                monkeypatch.delenv(k)

    # any VLLM_/SX_OPT_ switch, registered or not, invalidates
    assert changed(SX_OPT_MTP_ROWS="0")
    assert changed(SX_OPT_PIECEWISE_MIXED="0")
    assert changed(VLLM_SM70_NVFP4_MOE_TUNE_MAX_TOKENS="240")
    assert changed(VLLM_NEVER_HEARD_OF_THIS="1")
    assert changed(VLLM_REGISTERED="x") is False  # getter value, not env text

    # another build never shares a key
    fixed_identity["value"] = "build-B"
    assert ns["compile_factors"]() != factors
    fixed_identity["value"] = "build-A"
    assert ns["compile_factors"]() == factors

    # constants baked from runtime state
    sx.register_baked_constant("ple_rows:model.layers.0", (100, 20))
    with_rows = ns["compile_factors"]()
    assert with_rows != factors
    sx.register_baked_constant("ple_rows:model.layers.0", (90, 30))
    assert ns["compile_factors"]() != with_rows


# --------------------------------------------------------------------------
# lane block of VllmConfig.__post_init__
# --------------------------------------------------------------------------

_START_NEW = "# SX_OPT_COMPILE_CACHE (default off): reuse the torch.compile"
_START_OLD = 'if "VLLM_USE_AOT_COMPILE" not in os.environ:'
_END = 'self.compilation_config.inductor_compile_config["combo_kernels"] = True'


def _lane_block(source: str, start: str) -> str:
    lines = source.split("\n")
    first = next(i for i, line in enumerate(lines) if start in line)
    last = next(
        i for i, line in enumerate(lines) if i > first and _END in line
    )
    return textwrap.dedent("\n".join(lines[first:last]))


def _run_block(block: str, environ: dict, profiling: bool, glm: bool):
    log = boot.RecordingLogger()
    env_values = types.SimpleNamespace(
        VLLM_SM70_ALLOW_COMPILE_CACHE_FOR_PROFILING=profiling
    )
    namespace = {
        "os": types.SimpleNamespace(environ=environ),
        "logger": log,
        "envs": env_values,
        "sm70_glm5_dflash_tp8_pp1_verifier": glm,
        "_sx_compile_cache": sx,
    }
    # the real code runs with os.environ; the stub mapping must reach the
    # switch parser as well
    real = os.environ
    saved = dict(real)
    try:
        real.clear()
        real.update(environ)
        exec(compile(block, "<lane-block>", "exec"), namespace)  # noqa: S102
        environ.clear()
        environ.update(real)
    finally:
        real.clear()
        real.update(saved)
    return dict(environ), log.records


LANE_CASES = [
    dict(zip(("VLLM_USE_AOT_COMPILE", "VLLM_DISABLE_COMPILE_CACHE"), combo))
    for combo in itertools.product((None, "0", "1"), repeat=2)
]


def _case_env(case, switch=None):
    env = {k: v for k, v in case.items() if v is not None}
    if switch is not None:
        env["SX_OPT_COMPILE_CACHE"] = switch
    return env


@pytest.mark.parametrize("switch", [None, "0", "off"])
@pytest.mark.parametrize("profiling", [False, True])
@pytest.mark.parametrize("glm", [False, True])
@pytest.mark.parametrize("case", LANE_CASES)
def test_lane_block_off_is_byte_identical(case, profiling, glm, switch):
    old_source = boot.git_show(boot.BASE_REV, "vllm/config/vllm.py")
    if old_source is None:
        pytest.skip("git history with the base revision is not available")
    old_block = _lane_block(old_source, _START_OLD)
    new_block = _lane_block(boot.read(boot.CONFIG), _START_NEW)
    old = _run_block(old_block, _case_env(case), profiling, glm)
    new = _run_block(new_block, _case_env(case, switch), profiling, glm)
    old_env, new_env = old[0], new[0]
    new_env.pop("SX_OPT_COMPILE_CACHE", None)
    assert new_env == old_env
    assert new[1] == old[1]  # same log calls, same text, same order


@pytest.mark.parametrize("case", LANE_CASES)
def test_lane_block_on_never_forces_the_opt_out(case):
    block = _lane_block(boot.read(boot.CONFIG), _START_NEW)
    for switch, aot_default in (("1", "0"), ("aot", "1")):
        env, records = _run_block(block, _case_env(case, switch), False, False)
        # the opt-out is exactly what the user set, never written by the lane
        assert env.get("VLLM_DISABLE_COMPILE_CACHE") == case[
            "VLLM_DISABLE_COMPILE_CACHE"
        ]
        expect = case["VLLM_USE_AOT_COMPILE"] or aot_default
        assert env["VLLM_USE_AOT_COMPILE"] == expect
        assert records, "the decision is logged"
        # no leftover of the 0.0.3 quality-parity opt-out message
        assert not any("quality parity" in text for _, text in records)


# --------------------------------------------------------------------------
# build / checkpoint identity
# --------------------------------------------------------------------------


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def test_tree_digest_tracks_content_names_and_ignores_pycache(tmp_path):
    root = str(tmp_path / "pkg")
    _write(os.path.join(root, "a.py"), "x = 1\n")
    _write(os.path.join(root, "sub", "b.py"), "y = 2\n")
    first = sx.tree_digest(root)
    assert first == sx.tree_digest(root)
    _write(os.path.join(root, "__pycache__", "a.cpython-312.pyc"), "junk")
    _write(os.path.join(root, "notes.txt"), "not python")
    assert sx.tree_digest(root) == first
    _write(os.path.join(root, "sub", "b.py"), "y = 3\n")
    changed = sx.tree_digest(root)
    assert changed != first
    os.replace(os.path.join(root, "a.py"), os.path.join(root, "c.py"))
    assert sx.tree_digest(root) != changed  # a rename is a change


def test_native_fingerprint_tracks_size_and_content(tmp_path):
    root = str(tmp_path / "pkg")
    blob = os.path.join(root, "_C.abi3.so")
    _write(blob, "ELF" + "a" * 5000)
    first = sx.native_fingerprint(root)
    _write(blob, "ELF" + "b" * 5000)  # same size, other content
    second = sx.native_fingerprint(root)
    assert second != first
    _write(blob, "ELF" + "b" * 5001)  # other size
    assert sx.native_fingerprint(root) != second
    _write(os.path.join(root, "sub", "ops.pyd"), "x")
    assert sx.native_fingerprint(root) != sx.native_fingerprint(
        str(tmp_path / "pkg" / "sub")
    )


def test_identity_digest_is_stable_and_sensitive():
    base = {"torch": "2.10.0+cu128", "cuda": "12.8", "packages": {"vllm": {"py": "a"}}}
    assert sx.identity_digest(base) == sx.identity_digest(dict(reversed(base.items())))
    for key, value in (("torch", "2.10.1"), ("cuda", "12.9")):
        other = dict(base, **{key: value})
        assert sx.identity_digest(other) != sx.identity_digest(base)
    other = dict(base, packages={"vllm": {"py": "b"}})
    assert sx.identity_digest(other) != sx.identity_digest(base)


def test_env_identity_covers_inductor_triton_and_kernel_switches():
    env = {
        "TORCHINDUCTOR_MAX_AUTOTUNE": "1",
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/a",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TRITON_CACHE_DIR": "/cache/triton",
        "TRITON_CACHE_AUTOTUNING": "1",
        "Q_SCALE_CONSTANT": "200",
        "FLASH_QLA_SM70_USE_ORIGINAL_TILELANG": "1",
        "OMP_NUM_THREADS": "8",
        "HOME": "/cache/home",
    }
    got = sx.env_identity(env)
    assert got == {
        "FLASH_QLA_SM70_USE_ORIGINAL_TILELANG": "1",
        "Q_SCALE_CONSTANT": "200",
        "TORCHINDUCTOR_MAX_AUTOTUNE": "1",
        "TRITON_CACHE_AUTOTUNING": "1",
    }
    env["TORCHINDUCTOR_CACHE_DIR"] = "/tmp/other"
    assert sx.env_identity(env) == got  # directories do not matter


def test_gather_identity_fields_hash_the_installed_packages(tmp_path, monkeypatch):
    root = tmp_path / "vllm"
    _write(str(root / "__init__.py"), "")
    _write(str(root / "model.py"), "def forward(): ...\n")
    monkeypatch.setattr(sx, "_package_root", lambda name: str(root) if name == "vllm" else None)
    monkeypatch.setattr(sx, "_device_name", lambda: "Tesla V100-SXM2-32GB")
    fields = sx.gather_identity_fields({})
    assert set(fields["packages"]) == {"vllm"}
    digest = sx.identity_digest(fields)
    _write(str(root / "model.py"), "def forward(): return 1\n")
    assert sx.identity_digest(sx.gather_identity_fields({})) != digest
    monkeypatch.setattr(sx, "_device_name", lambda: "Tesla V100-PCIE-32GB")
    assert sx.identity_digest(sx.gather_identity_fields({})) != digest


def test_torch_backport_level_reads_the_installed_torch_file(monkeypatch, tmp_path):
    fake = tmp_path / "aot_compile_types.py"
    spec = types.SimpleNamespace(origin=str(fake))
    monkeypatch.setattr(sx.importlib.util, "find_spec", lambda name: spec)
    fake.write_text("class SerializableCallable: ...\n")
    assert sx.torch_backport_level() == "none"
    fake.write_text("def _serialize_triton_kernel(kernel): ...\n")
    assert sx.torch_backport_level() == "aot-triton-side-table"
    monkeypatch.setattr(sx.importlib.util, "find_spec", lambda name: None)
    assert sx.torch_backport_level() == "unknown"


def test_checkpoint_identity(tmp_path):
    model = tmp_path / "model"
    _write(str(model / "config.json"), '{"a": 1}')
    _write(str(model / "hf_quant_config.json"), '{"quant_algo": "NVFP4"}')
    shard = model / "model-00001-of-00002.safetensors"
    _write(str(shard), "w" * 100)
    os.utime(shard, ns=(10**18, 10**18))
    first = sx.checkpoint_identity(str(model))
    assert first == sx.checkpoint_identity(str(model))
    # a config edit that keeps size and mtime is still seen (json by content)
    config = model / "config.json"
    stat = config.stat()
    config.write_text('{"a": 2}')
    os.utime(config, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    after_config = sx.checkpoint_identity(str(model))
    assert after_config != first
    # a re-quantized shard of identical size: new mtime
    os.utime(shard, ns=(10**18 + 10**9, 10**18 + 10**9))
    after_shard = sx.checkpoint_identity(str(model))
    assert after_shard != after_config
    # another file of a different kind is irrelevant
    _write(str(model / "README.md"), "docs")
    assert sx.checkpoint_identity(str(model)) == after_shard
    # a hub id contributes only its name; a missing dir the same
    assert sx.checkpoint_identity("org/model") == "path:org/model"
    assert sx.checkpoint_identity(None) == "path:"


def test_counters_line_is_greppable():
    counters = types.SimpleNamespace(
        num_models_seen=2,
        num_graphs_seen=2,
        num_backend_compilations=5,
        num_cache_entries_updated=5,
        num_compiled_artifacts_saved=3,
        num_compiled_artifacts_loaded=0,
        num_aot_compiles=0,
        num_aot_artifacts_saved=0,
        num_aot_artifacts_loaded=0,
    )
    line = sx.counters_line(counters)
    assert "num_backend_compilations=5" in line
    assert "num_aot_artifacts_loaded=0" in line


def test_build_identity_is_computed_once_and_cached(monkeypatch):
    calls = []

    def fake_fields(environ=None):
        calls.append(1)
        return {"torch": "x"}

    sx.build_identity.cache_clear()
    monkeypatch.setattr(sx, "gather_identity_fields", fake_fields)
    first = sx.build_identity()
    assert sx.build_identity() == first and len(calls) == 1
    sx.build_identity.cache_clear()


def test_off_never_reads_the_filesystem_identity(monkeypatch):
    """With the switch off no identity (tree hash, NVML) is ever computed."""

    def boom(*args, **kwargs):
        raise AssertionError("identity computed with the switch off")

    monkeypatch.setattr(sx, "gather_identity_fields", boom)
    monkeypatch.setattr(sx, "build_identity", boom)
    assert sx.extra_compile_factors("off") == {}
    assert sx.extra_compile_factors() == {} or os.environ.get("SX_OPT_COMPILE_CACHE")
