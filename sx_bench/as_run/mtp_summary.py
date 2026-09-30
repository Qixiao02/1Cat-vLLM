"""mtp_summary.py <dir> <tag> : one SUMMARY line per benchmark cell of an MTP trial (see mtp_kv.sh)."""
import json
import os
import sys

w, tag = sys.argv[1:3]
for name in ("c1s", "c1g", "c4s2k", "c4g2k", "c4"):
    path = "%s/%s_%s.json" % (w, name, tag)
    if not os.path.exists(path):
        print("SUMMARY %-6s no result file" % name)
        continue
    for c in json.load(open(path))["cells"]:
        mem = max([g[1] for s in c["timeline"] if s.get("gpu") for g in s["gpu"]] or [0])
        sv = c.get("server", {})
        drafts = sv.get("spec_decode_num_drafts_total", 0)
        accepted = sv.get("spec_decode_num_accepted_tokens_total", 0)
        print("SUMMARY %-6s len %6d: prefill %s tok/s | decode %s/stream %s agg | tokens per round %.2f | peak GPU mem"
              " %5.0f MiB | kv peak %.1f%% | preempt %s%s" % (
                  name, c["length"], c.get("prefill_tok_s"), c.get("decode_tok_s"), c.get("decode_agg_tok_s"),
                  1 + accepted / drafts if drafts else 1.0, mem, 100 * c.get("kv_peak", 0), c.get("preemptions"),
                  " | ERRORS" if "errors" in c else ""))
