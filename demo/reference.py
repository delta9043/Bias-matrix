"""Python reference for the browser demo: the library and the experiments' classifier on the exported demo data.

    python -m demo.reference            # writes demo/reference.json
    node demo/check_core.js             # compares docs/assets/bm-core.js with it
"""
import base64
import json
from pathlib import Path

import numpy as np

import bias_matrix as bm
from experiments.baselines import reweighing_weights
from experiments.common import ROOT
from experiments.metrics import tpr_by_cell
from experiments.models import fit_linear

HERE = Path(__file__).resolve().parent
L2 = 1e-3


def load():
    """Dequantized embeddings and labels of docs/assets/demo-data.js (what the browser sees)."""
    text = (ROOT / "docs" / "assets" / "demo-data.js").read_text(encoding="utf-8")
    raw = json.loads(text[text.index("{"): text.rindex("}") + 1])
    d, scale = raw["meta"]["dim"], np.array(raw["scale"])

    def part(p):
        X = (np.frombuffer(base64.b64decode(p["x"]), dtype=np.int8).reshape(-1, d) * scale).astype(np.float32)
        return X, np.frombuffer(base64.b64decode(p["y"]), dtype=np.uint8).astype(int), np.frombuffer(base64.b64decode(p["z"]), dtype=np.uint8).astype(int)

    return part(raw["train"]), part(raw["test"]), len(raw["meta"]["classes"])


def summary(pred, y, z, C):
    tpr = tpr_by_cell(y, pred, z, C)
    gap = tpr[:, 1] - tpr[:, 0]
    return {"acc": float((pred == y).mean()), "gap_rms": float(np.sqrt((gap**2).mean())), "tpr": tpr.tolist()}


def main():
    (X, y, z), (Xt, yt, zt), C = load()
    out = {}
    base = fit_linear(X, y, C, l2=L2)
    out["none"] = summary(base.predict(Xt), yt, zt, C)

    means, counts = bm.cell_stats(X, y, z, C, 2)
    plan, q = bm.minimal_plan(counts)
    out["plan"] = {"q": float(q[1]), "additions": plan.tolist(), "total": int(plan.sum())}

    # BME: oblique and orthogonal edits; retrained and frozen models; edited test rows for a direct comparison
    for name, kw in (("bme_oblique", {}), ("bme_orth", {"oblique": False}), ("bme_oblique_k1", {"k": 1}), ("bme_oblique_k3", {"k": 3})):
        eraser = bm.bme(X, y, z, **kw)
        E, Et = eraser(X), eraser(Xt)
        out[name] = {"retrained": summary(fit_linear(E, y, C, l2=L2).predict(Et), yt, zt, C),
                     "frozen": summary(base.predict(Et), yt, zt, C), "edited_test_head": Et[:3, :6].tolist()}

    # BMA: several draws of the sources (the browser uses its own generator)
    runs = []
    for seed in range(5):
        src, target = bm.plan_sources(y, z, plan, np.random.default_rng(seed))
        extra = bm.mean_shift(X, y, z, src, target, means)
        runs.append(summary(fit_linear(np.concatenate([X, extra]), np.concatenate([y, y[src]]), C, l2=L2).predict(Xt), yt, zt, C))
    out["bma"] = {"acc": [r["acc"] for r in runs], "gap_rms": [r["gap_rms"] for r in runs]}
    out["reweighing"] = summary(fit_linear(X, y, C, l2=L2, weight=reweighing_weights(counts)[y, z]).predict(Xt), yt, zt, C)

    (HERE / "reference.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for name in ("none", "bme_oblique", "bme_orth", "bme_oblique_k1", "bme_oblique_k3"):
        r = out[name].get("retrained", out[name])
        print(f"{name:16s} acc {r['acc']:.4f} gap {r['gap_rms']:.4f}")
    print(f"bma (5 draws)    acc {np.mean(out['bma']['acc']):.4f} gap {np.mean(out['bma']['gap_rms']):.4f} | reweighing acc {out['reweighing']['acc']:.4f} "
          f"gap {out['reweighing']['gap_rms']:.4f} | plan {out['plan']['total']} additions, q {out['plan']['q']:.4f}")


if __name__ == "__main__":
    main()
