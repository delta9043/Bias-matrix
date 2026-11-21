"""LLM embedding extraction: mean-pooled hidden states, chunked and resumable.

    python -m experiments.embed --split train                 # Bias in Bios
    python -m experiments.embed --split train --kind cf       # gender-swapped text (CDA)
    python -m experiments.embed --split train --kind hazard --max-len 256 --batch 64 --chunk 4096
"""
import argparse
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .common import LAYERS, MODEL, emb_dir
from .data import load_bios, load_hazard, swap_gender
from .models import default_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--kind", default="orig", choices=["orig", "cf", "hazard"])  # cf: gender-swapped text; hazard: case title + body
    ap.add_argument("--limit", type=int, default=None)  # first N rows of the shuffled order
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--device", default=default_device())
    args = ap.parse_args()

    # Fixed Shuffled Order per Split (partial runs = uniform random subsamples)
    df = load_hazard() if args.kind == "hazard" else load_bios()
    df = df[df.split == args.split]
    order = np.random.default_rng(0).permutation(len(df))[: args.limit]
    df = df.iloc[order]
    texts = df["text" if args.kind == "hazard" else "hard_text"].tolist()
    if args.kind == "cf":
        texts = [swap_gender(t) for t in texts]
    uids = df.uid.to_numpy()

    out_dir = emb_dir(args.kind, args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    starts = [s for s in range(0, len(texts), args.chunk) if not (out_dir / f"chunk_{s // args.chunk:04d}.npz").exists()]
    print(f"{args.kind}_{args.split}: {len(texts)} texts, {len(starts)} chunks to do", flush=True)
    if not starts:
        return

    # Single Model Load; fp16 on a GPU, fp32 on CPU
    dtype = torch.float32 if args.device == "cpu" else torch.float16
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, dtype=dtype).to(args.device).eval()

    t0 = time.time()
    done = 0
    for s in starts:
        chunk_texts = texts[s : s + args.chunk]
        enc_len = [len(i) for i in tok(chunk_texts, truncation=True, max_length=args.max_len)["input_ids"]]
        by_len = np.argsort(enc_len)[::-1]  # length-sorted batches (less padding)
        emb = np.zeros((len(chunk_texts), len(LAYERS), model.config.hidden_size), dtype=np.float16)
        for b in range(0, len(by_len), args.batch):
            idx = by_len[b : b + args.batch]
            enc = tok([chunk_texts[i] for i in idx], padding=True, truncation=True,
                      max_length=args.max_len, return_tensors="pt").to(args.device)
            with torch.inference_mode():
                hs = model(**enc, output_hidden_states=True).hidden_states
            # Masked Mean Pooling in fp32 (fp16 sum overflow guard)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = torch.stack([(hs[l].float() * mask).sum(1) / mask.sum(1) for l in LAYERS], 1)
            emb[idx] = pooled.cpu().numpy().astype(np.float16)
        assert np.isfinite(emb).all()
        tmp = out_dir / "tmp_chunk.npz"
        np.savez(tmp, uid=uids[s : s + args.chunk], emb=emb)
        tmp.replace(out_dir / f"chunk_{s // args.chunk:04d}.npz")  # atomic rename (no partial chunks)
        done += len(chunk_texts)
        rate = done / (time.time() - t0)
        print(f"  chunk {s // args.chunk:04d} | {done}/{len(starts) * args.chunk} | {rate:.1f} texts/s "
              f"| max|emb| {np.abs(emb).max():.0f}", flush=True)


if __name__ == "__main__":
    main()
