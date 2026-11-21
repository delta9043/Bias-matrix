"""Model selection on the Bias in Bios dev split: embedding layer and L2 strength of the linear model -> select.json."""
import argparse

from . import models as M
from .common import LAYERS, load_emb, save_json
from .data import PROFESSIONS, load_bios

L2_GRID = (3e-4, 1e-3, 3e-3, 1e-2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=M.default_device())
    M.DEVICE = ap.parse_args().device
    y = load_bios().profession.to_numpy()
    rows = []
    for layer in LAYERS:
        (uid_tr, Xtr), (uid_dv, Xdv) = load_emb("orig", "train", layer), load_emb("orig", "dev", layer)
        for l2 in L2_GRID:
            acc = float((M.fit_linear(Xtr, y[uid_tr], len(PROFESSIONS), l2=l2).predict(Xdv) == y[uid_dv]).mean())
            rows.append({"layer": layer, "l2": l2, "dev_acc": acc})
            print(f"layer {layer:2d}  l2 {l2:.0e}  dev acc {acc:.4f}", flush=True)
    best = max(rows, key=lambda r: r["dev_acc"])
    print("selected:", best)
    save_json("select.json", {"rows": rows, "selected": best})


if __name__ == "__main__":
    main()
