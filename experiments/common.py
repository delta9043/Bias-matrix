"""Shared paths, constants, embedding and result I/O, figure style."""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("BIAS_MATRIX_DATA", ROOT / "data"))  # raw data, embeddings
RESULTS_DIR = Path(os.environ.get("BIAS_MATRIX_RESULTS", ROOT / "results"))  # result JSON, figures, tables
FIG_DIR = RESULTS_DIR / "figures"

MODEL = "Qwen/Qwen3-0.6B"
LAYERS = [7, 14, 21, 28]  # stored hidden_states indices (0 = token embedding output)


def selected():
    """Dev-selected (layer, l2) from select_layer; the reported values when it has not run."""
    path = RESULTS_DIR / "select.json"
    if path.exists():
        best = json.loads(path.read_text(encoding="utf-8"))["selected"]
        return best["layer"], best["l2"]
    return LAYERS[-1], 1e-3


# ------------------------------------------------------------------ embeddings
def emb_dir(kind, split):
    return DATA_DIR / "emb" / MODEL.split("/")[-1] / f"{kind}_{split}"


def load_emb(kind, split, layer):
    """Finished chunks of one layer: uids (n,), embeddings (n, d) float32."""
    chunks = [np.load(f) for f in sorted(emb_dir(kind, split).glob("chunk_*.npz"))]
    uid = np.concatenate([c["uid"] for c in chunks])
    X = np.concatenate([c["emb"][:, LAYERS.index(layer)] for c in chunks]).astype(np.float32)
    return uid, X


def load_uid(kind, split):
    """Row ids of the finished chunks (order of load_emb)."""
    return np.concatenate([np.load(f)["uid"] for f in sorted(emb_dir(kind, split).glob("chunk_*.npz"))])


def load_task(data, layer):
    """Train / test embeddings with class y and binary group z, and the number of classes.

    bios: occupation, gender (1 female).  hazard: item class, sales channel (1 remote sale).
    """
    from .data import load_bios, load_hazard

    if data == "bios":
        df, kind, cls, group = load_bios(), "orig", "profession", "gender"
    else:
        df, kind, cls, group = load_hazard(), "hazard", "cls", "channel"
    (uid_tr, Xtr), (uid_te, Xte) = load_emb(kind, "train", layer), load_emb(kind, "test", layer)
    y, z = df[cls].to_numpy(), df[group].to_numpy()
    return Xtr, y[uid_tr], z[uid_tr], Xte, y[uid_te], z[uid_te], int(max(y[uid_tr].max(), y[uid_te].max())) + 1


# --------------------------------------------------------------------- results
def save_json(name, obj):
    """Result JSON under the results directory (numpy types converted)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(type(o))

    path = RESULTS_DIR / name
    path.write_text(json.dumps(obj, indent=1, default=conv, ensure_ascii=False), encoding="utf-8")
    return path


def load_json(name):
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


# Figure Style: validated categorical order, recessive chrome, white surface
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"


def setup_style():
    import matplotlib as mpl
    from cycler import cycler

    mpl.use("Agg")
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.prop_cycle": cycler(color=PALETTE), "lines.linewidth": 2, "lines.markersize": 6.5,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "lines.markeredgecolor": SURFACE, "lines.markeredgewidth": 1.2,
        "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "legend.frameon": False, "legend.fontsize": 8, "savefig.dpi": 200, "savefig.bbox": "tight",
    })
