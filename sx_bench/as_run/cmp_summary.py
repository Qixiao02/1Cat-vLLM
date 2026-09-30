"""cmp_summary.py [arms...] : per-arm means over passes for the sweep, the long-input grid and the 128K request."""
import json, os, sys
W = "/mnt/2t/build/cmp1001"
arms = sys.argv[1:] or ["F1", "O1", "O2", "F2"]
def cells(path):
    return json.load(open(path))["cells"] if os.path.exists(path) else []
def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None
def f(x, d=1):
    return "-" if x is None else ("%.*f" % (d, x))
print("SWEEP 8K input, 256 greedy tokens: decode per stream / aggregate / prefill tok/s / last TTFT s")
print("arm  " + "  ".join("%-26s" % ("C%d" % c) for c in (1, 2, 4, 8, 16, 24)))
for a in arms:
    row = []
    for c in (1, 2, 4, 8, 16, 24):
        cs = cells("%s/%s_sweep_c%d.json" % (W, a, c))
        row.append("-" if not cs else "%s/%s/%s/%s" % (f(mean([x.get("decode_tok_s") for x in cs])), f(mean([x.get("decode_agg_tok_s") for x in cs]), 0), f(mean([x.get("prefill_tok_s") for x in cs]), 0), f(mean([x.get("ttft_last") for x in cs]))))
    print("%-4s " % a + "  ".join("%-26s" % r for r in row))
print("\nLONG C4 x length, 400 tokens: prefill / decode per stream / first TTFT / last TTFT / KV peak %")
for a in arms:
    cs = cells("%s/%s_long.json" % (W, a))
    out = []
    for L in (8000, 16000, 32000, 64000):
        g = [x for x in cs if x["length"] == L]
        if g:
            out.append("%dK %s/%s/%s/%s/%s%s" % (L // 1000, f(mean([x.get("prefill_tok_s") for x in g]), 0), f(mean([x.get("decode_tok_s") for x in g])), f(mean([x.get("ttft_first") for x in g])), f(mean([x.get("ttft_last") for x in g])), f(100 * max(x.get("kv_peak", 0) for x in g)), " wave%s" % g[0].get("wave") if g[0].get("wave") not in (None, 4) else ""))
    l = cells("%s/%s_l128k.json" % (W, a))
    print("%-4s %s | 128K C1: prefill %s decode %s" % (a, "  ".join(out), f(mean([x.get("prefill_tok_s") for x in l]), 0), f(mean([x.get("decode_tok_s") for x in l]))))
