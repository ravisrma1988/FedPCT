"""Per-client view of CCT20 baseline results: test size, unknown fraction, known-class coverage,
and whether the known-class shortfall is beyond binomial noise (one-sided 95%)."""
import json

import numpy as np

ROOT, TARGET = "${FEDPCT_ROOT}", 0.9
for s in range(5):
    pc = json.load(open(f"{ROOT}/results/cct20_perclient_seed{s}.json"))
    print(f"\nseed {s}  client: n_test unk_frac | known cov pooled / labelshift / local  (* = significantly < 0.90)")
    for i, r in enumerate(pc["pooled"]):
        n_known = r["n"] * (1 - r["unk_frac"])
        se = np.sqrt(TARGET * (1 - TARGET) / max(n_known, 1))
        cells = []
        for m in ["pooled", "labelshift", "local"]:
            c = pc[m][i]["cov_known"]
            cells.append(f"{c:.2f}{'*' if c < TARGET - 1.645 * se else ' '}")
        print(f"  {r['client']:>2}: {r['n']:>5} {r['unk_frac']:.2f} | " + " ".join(cells))
