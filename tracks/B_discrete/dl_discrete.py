"""Discrete choices in deep learning (PPT 1 slides 27-28) and the hinge loss (slide 4), on Track A's MLP (two moons).
  * magnitude pruning: keep the largest |w| (a discrete selection), accuracy vs sparsity
  * int8 quantisation: 4x smaller weights, accuracy change
  * architecture search: hidden width in {1,2,4,8,16,32} - exhaustive vs greedy "grow until no gain"
  * hinge loss (linear SVM by subgradient descent) vs logistic loss on the same features
  * L0 vs L1: exhaustive best-subset (L0-constrained, see experiments.py feature selection) vs Lasso path
Run: python dl_discrete.py -> results/dl_discrete.json"""
import json
import os
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "A_continuous"))
from optlib import adam  # noqa: E402
from problems import MLP  # noqa: E402


def train(h, seed, steps=1500):
    m = MLP(h=h, seed=seed)
    th = adam(m.batch_grad(32), m.theta0, steps, 0.03)[-1]
    return m, th


def prune(m, th, frac):
    """Zero the smallest-magnitude fraction of the weights (biases kept)."""
    W1, b1, W2, b2 = m.unpack(th)
    w = np.concatenate([W1.ravel(), W2])
    k = int(frac * len(w))
    if k == 0:
        return th
    thr = np.sort(np.abs(w))[k - 1]
    W1p = np.where(np.abs(W1) <= thr, 0, W1); W2p = np.where(np.abs(W2) <= thr, 0, W2)
    return np.concatenate([W1p.ravel(), b1, W2p, [b2]])


def finetune(m, th, steps=500):
    """Retrain only the surviving (non-zero) weights; pruned weights stay exactly zero (biases always trainable)."""
    W1, b1, W2, b2 = m.unpack(th)
    mask = np.concatenate([(W1 != 0).ravel(), np.ones_like(b1), W2 != 0, [1.0]]).astype(float)
    g = m.batch_grad(32)
    return adam(lambda x: g(x) * mask, th, steps, 0.01)[-1] * mask


def quantize_int8(th):
    """Symmetric per-tensor int8 quantisation and de-quantisation of all parameters."""
    s = np.abs(th).max() / 127
    return np.round(th / s).clip(-127, 127) * s


def main():
    out = {}
    seeds = range(5)
    pr = {}
    for frac in (0.0, 0.5, 0.8, 0.9, 0.95):
        accs, tuned = [], []
        for s in seeds:
            m, th = train(16, s)
            p = prune(m, th, frac)
            accs.append(m.accuracy(p))
            tuned.append(m.accuracy(finetune(m, p)))
        pr[f"{int(frac * 100)}% of weights removed"] = {"no_retraining_acc": st.mean(accs), "std": st.pstdev(accs),
                                                        "after_masked_finetune_acc": st.mean(tuned), "finetune_std": st.pstdev(tuned)}
    out["magnitude_pruning_h16"] = pr
    q = {"fp64": [], "int8": []}
    for s in seeds:
        m, th = train(16, s)
        q["fp64"].append(m.accuracy(th)); q["int8"].append(m.accuracy(quantize_int8(th)))
    out["int8_quantisation_h16"] = {k: {"test_acc_mean": st.mean(v), "std": st.pstdev(v)} for k, v in q.items()}
    out["int8_quantisation_h16"]["bytes_per_weight"] = {"fp32": 4, "int8": 1}
    widths = [1, 2, 4, 8, 16, 32]
    acc = {w: st.mean(train(w, s)[0].accuracy(train(w, s)[1]) for s in (0, 1, 2)) for w in widths}
    best = max(acc, key=acc.get)
    g, evals = widths[0], 1
    for w in widths[1:]:                                    # greedy: grow while validation accuracy improves
        evals += 1
        if acc[w] > acc[g] + 1e-9:
            g = w
        else:
            break
    out["architecture_search_width"] = {"accuracy_by_width": {str(k): v for k, v in acc.items()}, "exhaustive_best": best, "exhaustive_evaluations": len(widths),
                                        "greedy_choice": g, "greedy_evaluations": evals,
                                        "note": "accuracy is measured on the test split here for brevity; a real search would use a validation split"}
    hinge, logi = [], []
    for s in seeds:
        m = MLP(h=2, seed=s)
        X, y = m.Xtr, 2 * m.ytr - 1
        Xt, yt = m.Xte, 2 * m.yte - 1
        Xb, Xtb = np.c_[X, np.ones(len(X))], np.c_[Xt, np.ones(len(Xt))]
        w = np.zeros(3)
        for t in range(1, 2001):                            # linear SVM: hinge loss + L2, subgradient descent
            margin = y * (Xb @ w)
            g = -(Xb * y[:, None])[margin < 1].sum(0) / len(y) + 1e-3 * w
            w -= 0.5 / np.sqrt(t) * g
        hinge.append(float(np.mean(np.sign(Xtb @ w) == yt)))
        v = np.zeros(3)
        for _ in range(2000):                               # logistic regression, gradient descent
            p = 1 / (1 + np.exp(-(Xb @ v)))
            v -= 0.5 * Xb.T @ (p - (y + 1) / 2) / len(y)
        logi.append(float(np.mean(np.sign(Xtb @ v) == yt)))
    out["hinge_vs_logistic_linear_moons"] = {"hinge_svm_acc": st.mean(hinge), "logistic_acc": st.mean(logi)}
    json.dump(out, open(os.path.join(HERE, "results", "dl_discrete.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
