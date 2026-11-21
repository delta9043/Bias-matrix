"""Linear model on embeddings: multinomial logistic regression (torch, full-batch L-BFGS)."""
import gc

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cpu"  # set by the experiment scripts (--device)


def default_device():
    """Intel GPU (xpu) or CUDA when available, else CPU."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def free_memory():
    """Release cached tensors (an integrated GPU shares system RAM)."""
    gc.collect()
    if DEVICE == "xpu":
        torch.xpu.empty_cache()
    elif DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()


class LinearClassifier:
    """logits = x @ W + b in raw input coordinates (standardization folded in)."""

    def __init__(self, W, b):
        self.W, self.b = W, b

    def logits(self, X):
        return X @ self.W + self.b

    def predict(self, X):
        return self.logits(X).argmax(1)


def fit_linear(X, y, n_classes, l2=1e-4, weight=None, max_iter=1000):
    """Multinomial logistic regression: standardized inputs, L2 penalty, full-batch L-BFGS.

    weight: optional per-sample weights (reweighing).
    """
    Xt = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
    mean, std = Xt.mean(0), Xt.std(0) + 1e-6
    # In-Place on the Device Copy (halves peak memory); on CPU the tensor may alias the caller's array
    Xs = (Xt - mean) / std if DEVICE == "cpu" else Xt.sub_(mean).div_(std)
    del Xt
    yt = torch.as_tensor(y, dtype=torch.long, device=DEVICE)
    wt = None if weight is None else torch.as_tensor(weight / np.mean(weight), dtype=torch.float32, device=DEVICE)
    W = torch.zeros(X.shape[1], n_classes, device=DEVICE, requires_grad=True)
    b = torch.zeros(n_classes, device=DEVICE, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter, history_size=20,
                            tolerance_grad=1e-6, tolerance_change=1e-10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(Xs @ W + b, yt, reduction="none")
        loss = (loss * wt).mean() if wt is not None else loss.mean()
        loss = loss + 0.5 * l2 * (W**2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        W_raw = (W / std[:, None]).cpu().numpy()
        b_raw = (b - (mean / std) @ W).cpu().numpy()
    return LinearClassifier(W_raw, b_raw)
