"""Small plotting utilities for reliability diagrams.

`plot_reliability` writes a simple reliability diagram (accuracy vs. confidence)
to `save_path`. Kept deliberately minimal to avoid matplotlib dependencies in
unit tests other than this plotting helper.
"""

import matplotlib.pyplot as plt


def plot_reliability(bin_stats, save_path: str):
    """Plot reliability diagram (accuracy vs confidence) and save to file.

    bin_stats expects keys: 'acc', 'conf', 'count' and equal-length lists.
    """
    acc = bin_stats["acc"]; conf = bin_stats["conf"]
    bins = len(acc)
    xs = [(i+0.5)/bins for i in range(bins)]

    plt.figure()
    # reference diagonal y=x indicating perfect calibration
    plt.plot([0,1],[0,1], linestyle="--")
    # plot accuracy as filled bars and confidence as outlined bars for clarity
    plt.bar(xs, acc, width=1.0/bins, edgecolor="black", alpha=0.6, label="Accuracy")
    plt.bar(xs, conf, width=1.0/bins, edgecolor="black", fill=False, label="Confidence")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy / Confidence"); plt.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.close()
