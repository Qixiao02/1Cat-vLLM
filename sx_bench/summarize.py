"""Summarise detail_fork_c4.json and detail_official_c4.json (pfx_bench.py schema 2) into one side-by-side file.

    python summarize.py [DIR]      -> prints the table rows and writes summary_fork_vs_official.json into DIR
                                      (default: results/2026-09-30-fork-vs-official next to this file)
"""
import json
import os
import sys

HERE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "2026-09-30-fork-vs-official")
LANES = {"fork": "detail_fork_c4.json", "official": "detail_official_c4.json"}
ENGINE_TIMES = ("time_to_first_token_seconds", "request_queue_time_seconds", "request_prefill_time_seconds",
                "request_decode_time_seconds", "e2e_request_latency_seconds")


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def rnd(value, digits=1):
    return None if value is None else round(value, digits)


def lane_row(cells):
    requests = [q for c in cells for q in c["requests"]]
    gpu = [g for c in cells for s in c["timeline"] if s.get("gpu") for g in s["gpu"]]
    row = {
        "passes": len(cells),
        "prefill_tok_s": [c["prefill_tok_s"] for c in cells],
        "prefill_tok_s_mean": rnd(mean([c["prefill_tok_s"] for c in cells]), 0),
        "ttft_s_by_order": [rnd(mean([c["ttft"][i] for c in cells]), 2) for i in range(len(cells[0]["ttft"]))],
        "ttft_first_s": [c["ttft_first"] for c in cells],
        "ttft_last_s": [c["ttft_last"] for c in cells],
        "decode_tok_s": [c["decode_tok_s"] for c in cells],
        "decode_tok_s_mean": rnd(mean([c["decode_tok_s"] for c in cells]), 1),
        "decode_agg_tok_s": [c["decode_agg_tok_s"] for c in cells],
        "decode_agg_tok_s_mean": rnd(mean([c["decode_agg_tok_s"] for c in cells]), 0),
        "wave": [c["wave"] for c in cells],
        "wave_ttft_last_s": [c["wave_ttft_last"] for c in cells],
        "wave_prefill_tok_s": [c["wave_prefill_tok_s"] for c in cells],
        "wave_decode_tok_s": [c["wave_decode_tok_s"] for c in cells],
        "wave_decode_agg_tok_s": [c["wave_decode_agg_tok_s"] for c in cells],
        "e2e_s": [c["e2e_s"] for c in cells],
        "kv_peak_pct": rnd(100 * max(c["kv_peak"] for c in cells), 1),
        "waiting_peak": [c["waiting_peak"] for c in cells],
        "preemptions": [c["preemptions"] for c in cells],
        "cache_hit_tokens": [c["cache_hit_tokens"] for c in cells],
        "own_decode_tok_s_mean": rnd(mean([q["own_decode_tok_s"] for q in requests]), 1),
        "inter_token_ms": {
            "p50": rnd(mean([q["itl_ms"]["p50"] for q in requests]), 1),
            "p90": rnd(mean([q["itl_ms"]["p90"] for q in requests]), 1),
            "p99": rnd(mean([q["itl_ms"]["p99"] for q in requests]), 1),
            "max": rnd(max(q["itl_ms"]["max"] for q in requests), 1),
        },
        "gpu_util_pct_mean": rnd(mean([g[0] for g in gpu]), 1),
        "gpu_power_w_mean": rnd(mean([g[2] for g in gpu]), 1),
        "engine_mean_s": {},
    }
    for name in ENGINE_TIMES:
        total = sum(c["server"].get(name + "_sum", 0) for c in cells)
        count = sum(c["server"].get(name + "_count", 0) for c in cells)
        if count:
            row["engine_mean_s"][name] = rnd(total / count, 2)
    return row


def main():
    data = {lane: json.load(open(os.path.join(HERE, name))) for lane, name in LANES.items()}
    summary = {
        "test": "concurrency 4, cold random prompts (no shared prefix), 400 generated tokens per request, 2 passes; "
                "the same prompts on both lanes; one lane at a time",
        "lanes": {lane: {"engine": d["engine"], "started": d["started"], "finished": d["finished"], "args": d["args"]}
                  for lane, d in data.items()},
        "rows": [],
    }
    lengths = sorted({c["length"] for c in data["fork"]["cells"]})
    for length in lengths:
        row = {"length": length}
        for lane, d in data.items():
            row[lane] = lane_row([c for c in d["cells"] if c["length"] == length])
        summary["rows"].append(row)
        for lane in data:
            r = row[lane]
            print("%6d %-8s prefill %s mean %s | ttft by order %s | decode %s mean %s agg %s | wave %s wave-prefill %s "
                  "wave-decode %s wave-last %s | kv %s%% wait %s preempt %s hits %s | itl %s | own decode %s | gpu %s%% "
                  "%sW | engine %s | e2e %s" % (
                      length, lane, r["prefill_tok_s"], r["prefill_tok_s_mean"], r["ttft_s_by_order"], r["decode_tok_s"],
                      r["decode_tok_s_mean"], r["decode_agg_tok_s_mean"], r["wave"], r["wave_prefill_tok_s"],
                      r["wave_decode_tok_s"], r["wave_ttft_last_s"], r["kv_peak_pct"], r["waiting_peak"],
                      r["preemptions"], r["cache_hit_tokens"], r["inter_token_ms"], r["own_decode_tok_s_mean"],
                      r["gpu_util_pct_mean"], r["gpu_power_w_mean"], r["engine_mean_s"], r["e2e_s"]))
    for lane, d in data.items():
        print(lane, d["engine"]["version"], "blocks", d["engine"]["cache_config"].get("num_gpu_blocks"),
              d["started"], "->", d["finished"])
    json.dump(summary, open(os.path.join(HERE, "summary_fork_vs_official.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
