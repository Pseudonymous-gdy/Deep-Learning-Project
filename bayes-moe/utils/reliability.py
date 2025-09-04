import matplotlib.pyplot as plt

def plot_reliability(bin_stats, save_path: str):
    acc = bin_stats["acc"]; conf = bin_stats["conf"]
    bins = len(acc)
    xs = [(i+0.5)/bins for i in range(bins)]
    plt.figure()
    plt.plot([0,1],[0,1], linestyle="--")
    plt.bar(xs, acc, width=1.0/bins, edgecolor="black", alpha=0.6, label="Accuracy")
    plt.bar(xs, conf, width=1.0/bins, edgecolor="black", fill=False, label="Confidence")
    plt.xlabel("Confidence"); plt.ylabel("Accuracy / Confidence"); plt.legend()
    plt.tight_layout(); plt.savefig(save_path); plt.close()
