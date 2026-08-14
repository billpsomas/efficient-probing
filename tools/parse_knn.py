"""Collapse a k-NN sweep's raw stdout into one tidy result file.

main_linprobe.py --knn_eval prints a line per k; the job wrapper repeats that for
several temperatures. This picks the best (k, T) and writes a file in the same
spirit as the training logs: the full grid first, then the single number and the
setting that produced it, so the reported value is never separated from its config.
"""
import argparse, re, os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--variant", default="cls")
    a = ap.parse_args()

    if not os.path.exists(a.raw):
        raise SystemExit(f"{a.raw}: not found -- did the sweep run?")

    # The temperature marker is printed by main_linprobe itself so that it travels
    # through the same pipe as the results. An earlier version echoed it from the
    # shell, which left it out of the captured file and made every result unusable --
    # so fall back to the default rather than silently discarding everything.
    grid, T, feat = [], None, "raw"
    for line in open(a.raw, errors="replace"):
        m = re.search(r"=== features=(\w+) ===", line)
        if m:
            feat = m.group(1); continue
        m = re.search(r"=== T=([0-9.]+)", line)
        if m:
            T = float(m.group(1)); continue
        m = re.search(r"(\d+)-NN classifier result: Top1: ([0-9.]+), Top5: ([0-9.]+)", line)
        if m:
            grid.append((feat, T if T is not None else 0.07,
                         int(m.group(1)), float(m.group(2)), float(m.group(3))))

    if not grid:
        raise SystemExit(f"{a.raw}: no k-NN results found")

    best = max(grid, key=lambda g: g[3])
    with open(a.out, "w") as f:
        f.write("k-NN Evaluation\n")
        f.write(f"Model: {a.model}\n")
        f.write(f"Representation: {a.variant}\n")
        f.write("Protocol: features L2-normalised, cosine similarity, "
                "ImageNet-1k train as the database and val as queries\n")
        f.write("features: 'raw' is the backbone output as the probe sees it; "
                "'final_norm' additionally applies the encoder's final LayerNorm, which "
                "forward_features skips and which the LP head otherwise absorbs into its "
                "own learned BatchNorm\n")
        f.write("features, T, k, Top1, Top5\n")
        for fe, t, k, t1, t5 in grid:
            f.write(f"{fe}, {t}, {k}, {t1:.2f}, {t5:.2f}\n")
        f.write(f"\nBest: Top1 {best[3]:.2f}% with {best[0]} features at k={best[2]}, T={best[1]}\n")
        f.write(f"Top5: {best[4]:.2f}%\n")
    print(f"best k-NN: {best[3]:.2f}% ({best[0]}, k={best[2]}, T={best[1]}) -> {a.out}")


if __name__ == "__main__":
    main()
