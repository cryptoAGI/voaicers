#!/usr/bin/env python3
"""Record cryptoAGI/voaice tools/vprint.py on N metric sets, as the oracle for src/vclone.rs.

    python3 testing/vclone/make_oracle.py <path/to/vprint.py> [N] > tests/fixtures/vprint_oracle.jsonl

Values are written with repr(), which round-trips a double exactly, and cover the awkward cases on purpose: zero,
subnormals, values just under a power of ten, large metrics (a centroid in Hz), NaN and infinity (both encode as 0),
negatives, and plain random magnitudes. The first line names the vprint.py that produced the rest, by sha256.
"""
import hashlib, importlib.util, json, math, random, sys

path = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
spec = importlib.util.spec_from_file_location("vprint", path)
vp = importlib.util.module_from_spec(spec); spec.loader.exec_module(vp)
rng = random.Random(20261008)
SPECIAL = [0.0, -0.0, 5e-324, 1e-18, 9.999999999999999e-19, 1e-17, 0.1, 0.7, 1.1, 0.9999999999999999, 1.0,
           22050.0, 11025.0, 96.8048780487805, 1e6, 123456789.123456789, -1.5, float("nan"), float("inf"), -float("inf")]

def one():
    r = rng.random()
    if r < 0.25:
        return rng.choice(SPECIAL)
    if r < 0.5:
        return rng.random()
    if r < 0.75:
        return rng.uniform(0, 12000)
    return 10 ** rng.uniform(-20, 7)

print(json.dumps({"oracle": "vprint.py", "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(), "cases": n}))
for _ in range(n):
    vals = [one() for _ in vp.METRICS]
    p = vp.vprint(dict(zip(vp.METRICS, vals)))
    print(json.dumps({"in": [repr(v) for v in vals], "hash": p["hash"], "hash512": p["hash512"],
                      "uint256": p["uint256"], "canonical": p["canonical"]}))
