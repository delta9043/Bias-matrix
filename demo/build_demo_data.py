"""Data of the browser demo (docs/assets/demo-data.js): seven medical occupations of Bias in Bios,
PCA-reduced embeddings of a training sample and of the whole test split, and the example biographies.

    python -m demo.build_demo_data        # needs the Bias in Bios embeddings (run_embed.sh) and the LLM
"""
import base64
import json
from pathlib import Path

import numpy as np

from bias_matrix import covariance
from experiments.common import LAYERS, MODEL, ROOT, load_task, selected
from experiments.data import PROFESSIONS, swap_gender

HERE = Path(__file__).resolve().parent
OUT = ROOT / "docs" / "assets" / "demo-data.js"
CLASSES = ["surgeon", "chiropractor", "dentist", "physician", "psychologist", "nurse", "dietitian"]  # by female share
DIM, N_TRAIN, MAX_LEN = 64, 8000, 192


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def embed(texts, layer):
    """Mean-pooled hidden states of the frozen LLM, one text at a time (CPU, fp32; same pooling as experiments.embed)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=torch.float32).eval()
    out = []
    for t in texts:
        enc = tok([t], truncation=True, max_length=MAX_LEN, return_tensors="pt")
        with torch.inference_mode():
            hs = model(**enc, output_hidden_states=True).hidden_states[layer]
        mask = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((hs.float() * mask).sum(1) / mask.sum(1))[0].numpy())
    return np.stack(out)


def main():
    layer = selected()[0]
    assert layer in LAYERS
    Xtr, ytr, ztr, Xte, yte, zte, _ = load_task("bios", layer)
    remap = np.full(len(PROFESSIONS), -1)
    remap[[PROFESSIONS.index(p) for p in CLASSES]] = np.arange(len(CLASSES))

    # Occupation Subset; Random Training Sample; Whole Test Split of the Subset
    tr, te = np.flatnonzero(remap[ytr] >= 0), np.flatnonzero(remap[yte] >= 0)
    tr = tr[np.random.default_rng(0).permutation(len(tr))[:N_TRAIN]]
    Xa, ya, za = Xtr[tr], remap[ytr[tr]], ztr[tr]
    Xb, yb, zb = Xte[te], remap[yte[te]], zte[te]

    # PCA Basis from the Training Sample Only
    mean, cov = covariance(Xa)
    w, V = np.linalg.eigh(cov)
    P = V[:, ::-1][:, :DIM]
    kept = float(w[::-1][:DIM].sum() / w.sum())
    Za, Zb = (Xa - mean) @ P, (Xb - mean) @ P

    # Int8 Quantization with One Scale per Dimension
    scale = np.abs(np.concatenate([Za, Zb])).max(0) / 127

    def quantize(Z):
        return np.clip(np.round(Z / scale), -127, 127).astype(np.int8)

    # Example Biographies: female version as written, male version by the gender swap of the CDA baseline
    spec = json.loads((HERE / "examples.json").read_text(encoding="utf-8"))["examples"]
    texts = [(e["text"], swap_gender(e["text"])) for e in spec]
    E = (embed([t for pair in texts for t in pair], layer) - mean) @ P
    examples = [{"id": e["id"], "label": None if e["label"] is None else CLASSES.index(e["label"]), "text": {"f": tf, "m": tm},
                 "x": {"f": np.round(E[2 * i], 4).tolist(), "m": np.round(E[2 * i + 1], 4).tolist()}}
                for i, (e, (tf, tm)) in enumerate(zip(spec, texts))]

    data = {
        "meta": {"classes": CLASSES, "groups": ["male", "female"], "dim": DIM, "n_train": int(len(Za)), "n_test": int(len(Zb)),
                 "variance_kept": round(kept, 4), "model": MODEL, "layer": layer},
        "scale": np.round(scale, 6).tolist(),
        "train": {"x": b64(quantize(Za)), "y": b64(ya.astype(np.uint8)), "z": b64(za.astype(np.uint8))},
        "test": {"x": b64(quantize(Zb)), "y": b64(yb.astype(np.uint8)), "z": b64(zb.astype(np.uint8))},
        "examples": examples,
    }
    OUT.write_text("window.BM_DEMO_DATA = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"{OUT.name}: train {Za.shape} test {Zb.shape} | variance kept {kept:.3f} | {OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
